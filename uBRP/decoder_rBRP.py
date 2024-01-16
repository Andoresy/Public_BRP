import torch
import torch.nn as nn

from encoder import MultiHeadAttention, ScaledDotProductAttention, GraphAttentionEncoder, SingleHeadAttention
from Env import Env
from sampler import TopKSampler, CategoricalSampler
from data import generate_data
class Decoder_rBRP(nn.Module):
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
                 **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.embed_dim = embed_dim
        self.Encoder = GraphAttentionEncoder(n_heads, embed_dim, n_encode_layers, max_stacks, max_tiers, n_containers, ff_hidden=ff_hidden)
        self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wq_step = nn.Linear(embed_dim, embed_dim, bias=False)
        self.WO = nn.Linear(embed_dim, embed_dim, bias=False)
        # node embedding은 multi-head attention과 방문할 노드를 결정할 때 사용됩니다.
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk_1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk_2 = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.MHA = MultiHeadAttention(n_heads, embed_dim, False)
        self.SHA = SingleHeadAttention(clip=10, head_depth = embed_dim)

    def compute_static(self, node_embeddings, graph_embedding):
        self.Q_fixed = self.Wq_fixed(graph_embedding[:, None, :])
        self.K1 = self.Wk_1(node_embeddings)
        self.V = self.Wv(node_embeddings)
        self.K2 = self.Wk_2(node_embeddings)

    def compute_dynamic(self, mask, step_context):
        Q_step = self.Wq_step(step_context)
        Q1 = self.Q_fixed + Q_step
        Q2 = self.MHA([Q1, self.K1, self.V], mask=mask)
        Q2 = self.WO(Q2)
        logits = self.SHA([Q2, self.K2, None], mask=mask)
        return logits.squeeze(dim=1)
    def forward(self, x, n_containers=8, return_pi=False, decode_type='sampling'):

        env = Env(self.device,x,self.embed_dim)

        #先清理已经满足的
        env.clear()
        print('-----------------------')
        encoder_output=self.Encoder(env.x)
        node_embeddings, graph_embedding = encoder_output
        env.node_embeddings=node_embeddings
        self.compute_static(node_embeddings, graph_embedding)

        #mask(batch,max_stacks,1) 
        #step_context=target_stack_embedding(batch, 1, embed_dim) 
        mask, step_context = env._create_t1()

        #default n_samples=1
        selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
        log_ps, tours = [], []
        print(n_containers)
        batch,max_stacks,max_tiers = x.size()
        cost=torch.zeros(batch).to(self.device)
        ll=torch.zeros(batch).to(self.device)

        for i in range(n_containers * max_tiers):

            # logits (batch,max_stacks)
            logits = self.compute_dynamic(mask, step_context)
            # log_p (batch,max_stacks)
            log_p = torch.log_softmax(logits, dim=-1)
            # next_node (batch,1)
            next_node = selecter(log_p)
            cost += (1.0 - env.empty.type(torch.float64))
            ll += torch.gather(input=log_p,dim=1,index=next_node).squeeze(-1)

            #solv the actions
            env._get_step(next_node)

            if env.all_empty():
                break

            # re-compute node_embeddings
            encoder_output = self.Encoder(env.x)
            node_embeddings, graph_embedding = encoder_output
            env.node_embeddings = node_embeddings
            self.compute_static(node_embeddings, graph_embedding)

            mask, step_context = env._create_t1()


        return cost, ll


if __name__ == '__main__':
    batch, n_nodes, embed_dim = 1, 21, 128
    data = generate_data(device = 'cpu', n_samples=batch)
    decoder = Decoder_rBRP('cpu', embed_dim, n_heads=8, clip=10.)
    node_embeddings = torch.rand((batch, n_nodes, embed_dim), dtype=torch.float).to('cpu')
    graph_embedding = torch.rand((batch, embed_dim), dtype=torch.float).to('cpu')
    encoder_output = (node_embeddings, graph_embedding)
    decoder.train()
    cost, ll= decoder(data, return_pi=True, decode_type='sampling')
    print('\ncost: ', cost.size(), cost)
    print('\nll: ', ll.size(), ll)
