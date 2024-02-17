import torch
import torch.nn as nn
import math
from encoder_LSTM import MultiHeadAttention, ScaledDotProductAttention, GraphAttentionEncoder, SingleHeadAttention
from Env_V3 import Env
from sampler import TopKSampler, CategoricalSampler, New_Sampler
from data import generate_data
from decoder_utils import concat_embedding, concat_graph_embedding
class Decoder_uBRP(nn.Module):
    def __init__(self, 
                 device, 
                 embed_dim=128, 
                 n_encode_layers=3, 
                 n_heads=8, 
                 clip=10., 
                 ff_hidden = 512, 
                 n_containers=8, 
                 max_stacks = 4,
                 max_tiers = 4,
                 return_pi = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.embed_dim = embed_dim
        self.concat_embed_dim = embed_dim*2
        self.total_embed_dim = embed_dim*3
        self.return_pi = return_pi
        self.Encoder = GraphAttentionEncoder(n_heads, embed_dim, n_encode_layers, max_stacks, max_tiers, n_containers, ff_hidden=ff_hidden).to(device)
        self.Wk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wout = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_step = nn.Linear(embed_dim , embed_dim, bias=False)
        
        self.MHA = MultiHeadAttention(n_heads, embed_dim, False)
        self.SHA = SingleHeadAttention(clip=clip, head_depth=embed_dim)
    def compute_static(self, node_embeddings, graph_embedding):
        self.Q_fixed = self.Wq_fixed(graph_embedding[:, None, :])
        self.K1 = self.Wk1(node_embeddings)
        self.V = self.Wv(node_embeddings)
        self.K2 = self.Wk2(node_embeddings)

    def compute_dynamic(self, mask, step_context):
        Q_step = self.Wq_step(step_context)
        Q1 = self.Q_fixed + Q_step
        Q2 = self.MHA([Q1, self.K1, self.V], mask=mask)
        Q2 = self.Wout(Q2)
        logits = self.SHA([Q2, self.K2, None], mask=mask)
        return logits.squeeze(dim=1)
    def forward(self, x, n_containers=8, return_pi=False, decode_type='sampling'):

        batch,max_stacks,max_tiers = x.size()
        env = Env(self.device,x,self.concat_embed_dim)
        env.clear()
        encoder_output=self.Encoder(env.x)
        node_embeddings, graph_embedding = encoder_output
        #concated_embeddings = concat_embedding(node_embeddings)
        self.compute_static(node_embeddings, graph_embedding)
        mask = env.create_mask_uBRP().view(batch, max_stacks, max_stacks)

        #default n_samples=1
        selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler(), 'new_sampling': New_Sampler()}.get(decode_type, None)
        log_ps, tours = [], []
        cost=torch.zeros(batch).to(self.device)
        Length = torch.zeros(batch).to(self.device)
        ll=torch.zeros(batch).to(self.device)

        for i in range(n_containers * max_tiers):

            # logits (batch,max_stacks)
            logits = self.compute_dynamic(mask, node_embeddings).view(batch, max_stacks*max_stacks)
            # log_p (batch,max_stacks)
            log_p = torch.log_softmax(logits, dim=1)
            # next_node (batch,1)
            if decode_type == 'new_sampling':
                next_action = selecter(logits)
            else:
                next_action = selecter(log_p)
            source_node, dest_node = next_action%max_stacks, next_action//max_stacks
            actions = torch.cat((source_node,dest_node), 1)
#            DEBUGGING Print
            if(return_pi):
                print('------------------------------------')
                print('env:', env.x)
    #            print('mask:', mask.view(batch, max_stacks, max_stacks))
                print('p:',torch.softmax(logits, dim=1).view(batch, max_stacks, max_stacks))
                print("action:", actions)
            cost += (1.0 - env.empty.type(torch.float64))
            Length += (1.0 - env.empty.type(torch.float64))
            #만약 필요하다면 끝난 node들에 대해 더해지는 일은 없어야할듯
            temp_log_p = log_p.clone()#수정필요
            temp_log_p[env.empty, :] = 0#수정필요
#            print(temp_log_p)
            #----수정필요
            ll += torch.gather(input=temp_log_p,dim=1,index=next_action).squeeze(-1)
            #solv the actions
            env.step(actions)
            #cost -= env.last_retrieved_nums.type(torch.float64) * 0.1
            if env.all_empty():
                break

            # re-compute node_embeddings
            encoder_output=self.Encoder(env.x)
            node_embeddings, graph_embedding = encoder_output
            #concated_embeddings = concat_embedding(node_embeddings)
            self.compute_static(node_embeddings, graph_embedding)
            mask = env.create_mask_uBRP().view(batch, max_stacks, max_stacks)
#        print('\ncost(Number of Relocations):\n', cost)
#        print('\nll(Sum of Log Probabilities on trajectory):\n', ll)

        return cost, ll, Length


if __name__ == '__main__':
    batch, max_stacks, embed_dim = 1, 4, 128
    data = generate_data(device = 'cuda:0', n_samples=batch, max_stacks=max_stacks)
    decoder = Decoder_uBRP('cuda:0', embed_dim, max_stacks=max_stacks, n_heads=8, clip=10.).to('cuda:0')
    decoder.train()
    cost, ll, Length= decoder(data, return_pi=True, decode_type='sampling')
    print('\nbatch, max_stacks, embed_dim',batch, max_stacks, embed_dim)
    print('\ncost:\n', cost)
    print('\ncost(Number of Length):\n', Length)
    print('\nll(Sum of Log Probabilities on trajectory):\n', ll)
