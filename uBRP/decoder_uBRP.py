import torch
import torch.nn as nn
import math
from encoder import MultiHeadAttention, ScaledDotProductAttention, GraphAttentionEncoder, SingleHeadAttention
from Env_V2 import Env
from sampler import TopKSampler, CategoricalSampler
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
        self.Wq_fixed = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.Wk_2 = nn.Linear(embed_dim*3, embed_dim*3, bias=False)

        self.W_O = nn.Sequential(
            nn.Linear(embed_dim*3, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2,embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1)
            #nn.Linear(256, 512), # It can be More Deeper
            #nn.ReLU(),
            #nn.Linear(512,256),
            #nn.ReLU(),
            #nn.Linear(256,128),
            #nn.ReLU()
        ).to(device)
        
        self.MHA = MultiHeadAttention(n_heads, embed_dim*2, False)

    def compute_static(self, node_embeddings, graph_embedding):
        pass
        #self.Q_fixed = self.Wq_fixed(graph_embedding[:, None, :])
        #self.K = self.Wk_2(node_embeddings)

    def compute_dynamic(self, mask, node_embeddings):
        logits = self.W_O(node_embeddings)
        logtis_with_mask = logits - mask.to(torch.int)*1e9
        return logtis_with_mask.squeeze(dim=2)
    def forward(self, x, n_containers=8, return_pi=False, decode_type='sampling'):

        env = Env(self.device,x,self.concat_embed_dim)
        env.clear()
        encoder_output=self.Encoder(env.x)
        node_embeddings, graph_embedding = encoder_output
        concat_node_embeddings = concat_embedding(node_embeddings, device = self.device)
        total_embeddings = concat_graph_embedding(graph_embedding, concat_node_embeddings)
        #mask(batch,max_stacks,1) 
        #step_context=target_stack_embedding(batch, 1, embed_dim) 
        mask = env.create_mask_uBRP()

        #default n_samples=1
        selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
        log_ps, tours = [], []
        batch,max_stacks,max_tiers = x.size()
        cost=torch.zeros(batch).to(self.device)
        ll=torch.zeros(batch).to(self.device)

        for i in range(n_containers * max_tiers):

            # logits (batch,max_stacks)
            logits = self.compute_dynamic(mask, total_embeddings)
            # log_p (batch,max_stacks)

            log_p = torch.log_softmax(logits, dim=1)

            # next_node (batch,1)
            next_action = selecter(log_p)
            source_node, dest_node = next_action//max_stacks, next_action%max_stacks
            actions = torch.cat((source_node,dest_node), 1)
#            DEBUGGING Print
            if(return_pi):
                print('------------------------------------')
                print('env:', env.x)
    #            print('mask:', mask.view(batch, max_stacks, max_stacks))
    #            print('log_p:',log_p.view(batch, max_stacks, max_stacks))
                print("action:", actions)
            cost += (1.0 - env.empty.type(torch.float64))
            #만약 필요하다면 끝난 node들에 대해 더해지는 일은 없어야할듯
            temp_log_p = log_p.clone()#수정필요
            temp_log_p[env.empty, :] = 0#수정필요
#            print(temp_log_p)
            #----수정필요
            ll += torch.gather(input=temp_log_p,dim=1,index=next_action).squeeze(-1)

            #solv the actions
            env.step(actions)

            if env.all_empty():
                break

            # re-compute node_embeddings
            encoder_output = self.Encoder(env.x)
            node_embeddings, graph_embedding = encoder_output
            concat_node_embeddings = concat_embedding(node_embeddings, device= self.device)
            total_embeddings = concat_graph_embedding(graph_embedding, concat_node_embeddings)

            mask = env.create_mask_uBRP()
#        print('\ncost(Number of Relocations):\n', cost)
#        print('\nll(Sum of Log Probabilities on trajectory):\n', ll)

        return cost, ll


if __name__ == '__main__':
    batch, max_stacks, embed_dim = 32, 4, 128
    data = generate_data(device = 'cuda:0', n_samples=batch, max_stacks=max_stacks)
    decoder = Decoder_uBRP('cuda:0', embed_dim, max_stacks=max_stacks, n_heads=8, clip=10.)
    decoder.train()
    cost, ll= decoder(data, return_pi=True, decode_type='sampling')
    print('\nbatch, max_stacks, embed_dim',batch, max_stacks, embed_dim)
    print('\ncost(Number of Relocations):\n', cost)
    print('\nll(Sum of Log Probabilities on trajectory):\n', ll)
