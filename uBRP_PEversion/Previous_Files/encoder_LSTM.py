import torch
import torch.nn as nn
import torch.nn.functional as F
import math
""" https://github.com/wouterkool/attention-learn-to-route
    TSP Attention Code (Given)
    참고하였습니다.
"""
class SkipConnection(nn.Module):
    """ SkipConnection 구현 
        (Q,K,V,mask) -> (Q+output), attn_dist
    """
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module
    def forward(self, input):
        output= self.module(input)
        return input + output

class Normalization(nn.Module): #wounterkool git 참고
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class ScaledDotProductAttention(nn.Module): #Tsp_Attention.ipynb
    """ Attention(Q,K,V) = softmax(QK^T/root(d_k))V
    """
    def __init__(self, d_k):# d_k: head를 나눈 이후의 key dimension
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k # key dimenstion
        self.inf = 1e9
        #self.dropout = nn.Dropout(0.5)
    def forward(self, Q, K, V, mask):
        # key, query의 곱을 통해 attention weight를 계산하고 value의 weighted sum인 output을 생성
        # input: Q, K, V, mask (query, key, value, padding 및 시간을 반영하기 위한 mask)
        # output: output, attn_dist (value의 weight sum, attention weight)
        # dim of Q,K,V: batchSize x n_heads x seqLen x d_k(d_v)
        d_k = self.d_k        
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d_k) 
                    # dim of attn_score: batchSize x n_heads x seqLen_Q x seqLen_K
                    #wj) batch matrix multiplication
        if mask is None:
            mask = torch.zeros_like(attn_score).bool()
        else:
            attn_score = attn_score.masked_fill(mask[:, None, None, :, 0].repeat(1, attn_score.size(1), 1, 1) == True, -self.inf)

        attn_dist = F.softmax(attn_score, dim=-1)  # attention distribution
        output = torch.matmul(attn_dist, V)  # dim of output : batchSize x n_heads x seqLen x d_v

        return output, attn_dist

class SingleHeadAttention(nn.Module):
    def __init__(self, clip=10, head_depth=16, inf=1e+10, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip
        self.inf = inf
        self.scale = math.sqrt(head_depth)

    # self.tanh = nn.Tanh()

    def forward(self, x, mask=None):
        """ Q: (batch, n_heads, q_seq(=max_stacks or =1), head_depth)
            K: (batch, n_heads, k_seq(=max_stacks), head_depth)
            logits: (batch, n_heads, q_seq(this could be 1), k_seq)
            mask: (batch, max_stacks, 1), e.g. tf.Tensor([[ True], [ True], [False]])
            mask[:,None,None,:,0]: (batch, 1, 1, stacks) ==> broadcast depending on logits shape
            [True] -> [1 * -np.inf], [False] -> [logits]
            K.transpose(-1,-2).size() == K.permute(0,1,-1,-2).size()
        """
        Q, K, V = x
        logits = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        logits = self.clip * torch.tanh(logits)

        if mask is not None:
            return logits.masked_fill(mask.permute(0, 2, 1) == True, -self.inf)
        return logits
class MultiHeadAttention(nn.Module):
    """ Skip_Connection 은 Built_in 되어있습니다.
        Norm은 따로 진행합니다. (tsp_attention.ipynb와 다름)
    """
    def __init__(self, n_heads, embed_dim, is_encoder):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.d_k = embed_dim//n_heads
        self.d_v = embed_dim//n_heads
        
        assert self.embed_dim % self.n_heads == 0 #embed_dim = n_heads * head_depth

        self.is_encoder = is_encoder
        if self.is_encoder:
            self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
            self.layerNorm = nn.LayerNorm(embed_dim, 1e-6) # layer normalization
        self.attention = ScaledDotProductAttention(self.d_k)
        #self.init_parameters()
    def init_parameters(self):
        #JK: 필요하다면 구현해야함
        pass
    def forward(self, x, mask=None):
        Q,K,V = x
        batchSize, seqLen_Q, seqLen_K = Q.size(0), Q.size(1), K.size(1) # decoder의 경우 query와 key의 length가 다를 수 있음
        residual = Q
        # Query, Key, Value를 (n_heads)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        # dim : batchSize x seqLen x embed_dim -> batchSize x seqLen x n_heads x d_k
        if self.is_encoder:
            Q = self.W_Q(Q)
            K = self.W_K(K)
            V = self.W_V(V)
        
        Q = Q.view(batchSize, seqLen_Q, self.n_heads, self.d_k)
        K = K.view(batchSize, seqLen_K, self.n_heads, self.d_k)
        V = V.view(batchSize, seqLen_K, self.n_heads, self.d_v)
        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # dim : batchSize x seqLen x n_heads x d_k -> batchSize x n_heads x seqLen x d_k
        output, attn_dist = self.attention(Q, K, V, mask)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous()  # dim : batchSize x n_heads x seqLen x d_k -> batchSize x seqLen x n_heads x d_k
        output = output.view(batchSize, seqLen_Q, -1)  # dim : batchSize x seqLen x n_heads x d_k -> batchSize x seqLen x d_model

        # Linear Projection, Residual sum
        if self.is_encoder:
            output = residual + self.W_O(output)
        
        return output

class MultiHeadAttentionLayer(nn.Module): #Self-Attention
    """ h_ = BN(h+MHA(h))
        h = BN(h_ + FF(h_))
    """
    def __init__(self, n_heads, embed_dim, ff_hidden = 512, normalization = 'instance', is_encoder=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.MHA = MultiHeadAttention(n_heads, embed_dim=embed_dim, is_encoder=is_encoder) #Maybe Modified
        self.BN1 = Normalization(embed_dim, normalization)
        self.BN2 = Normalization(embed_dim, normalization)
        
        self.FF_sub = SkipConnection(
                        nn.Sequential(
                            nn.Linear(embed_dim, ff_hidden), #bias = True by default
                            nn.ReLU(),
                            nn.Linear(ff_hidden, embed_dim)  #bias = True by default
                        )
                    )
    def forward(self, x, mask=None):
        #######################################
        #With BatchNorm/InstanceNorm
        x = [x,x,x] # Self_Attention
        x = self.BN1(self.MHA(x, mask=mask))
        x = self.BN2(self.FF_sub(x))
        #######################################
        #######################################
        #Without BatchNorm
        #x = [x,x,x]
        #x = self.FF_sub(self.MHA(x, mask=mask))    
        #######################################    
        return x

class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads = 8, embed_dim=32, n_layers=3, max_stacks=4, max_tiers=4,  n_containers = 8, normalization = 'instance', ff_hidden = 512, LSTM_num_layers = 1):
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.max_stacks = max_stacks
        self.max_tiers = max_tiers
        self.LSTM_num_layers = LSTM_num_layers
        self.LSTM = nn.LSTM(input_size=embed_dim, hidden_size = embed_dim, batch_first = True, num_layers = LSTM_num_layers)
        self.LSTM_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.init_positional_encoding = nn.Sequential(
            nn.Linear(1, 16, bias=True),
            nn.ReLU(),
            #nn.Dropout(.5),
            nn.Linear(16, 1, bias = True)
        )
        #self.GRU = nn.GRU(input_size = embed_dim, hidden_size=embed_dim, batch_first = True)
        self.init_block_embed = nn.Sequential(
            nn.Linear(2, embed_dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim, bias=True)
        )
        self.encoder_layers = nn.ModuleList([MultiHeadAttentionLayer(n_heads, embed_dim,ff_hidden ) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        """ x(batch, max_stacks, max_tiers)
            return: (node_embeddings, graph_embedding)
            =((batch, max_stacks, embed_dim), (batch, embed_dim))
        """
        batch,max_stacks,max_tiers = x.size()
        x = x.clone()
        #x = x + torch.normal(.0, 0.002, size=(batch,max_stacks,max_tiers)).to('cuda:0') #Noise Layer
        x = x.view(batch, max_stacks, max_tiers, 1)
        positional_encoding = self.init_positional_encoding(torch.linspace(0,1,max_tiers).repeat(batch, max_stacks, 1).unsqueeze(-1).to('cuda:0')) # Can be Changed To Learnable
        x = torch.cat([x, positional_encoding], dim=3)
        #print(x.size())
        x = self.init_block_embed(x)
        x = x.view(batch*max_stacks, max_tiers, self.embed_dim)
        _, (hidden_state, _) = self.LSTM(x)
        #_, hidden_state = self.GRU(x)
        h = hidden_state[self.LSTM_num_layers-1,:,:].view(batch, max_stacks, self.embed_dim) # 제일 끝의 출력값 사용
        x = self.LSTM_embed(h)

        #x = self.init_embed(x)[0] ##LSTM!

        for layer in self.encoder_layers:
            x = layer(x, mask)
        return (x, torch.mean(x, dim=1))

if __name__ == "__main__":
    batch, n_nodes, embed_dim = 5, 21, 128
    max_stacks, max_tiers, n_containers = 4, 4, 8
    device = 'cuda:0'
    encoder = GraphAttentionEncoder().to(device)
    data = torch.randn((batch, 4, 4), dtype=torch.float).to(device)
    output = encoder(data, mask=None)
    print(f"output[0] shape: {output[0].shape}")
    print(f"output[1] shape: {output[1].shape}")
    cnt = 0
    for i, k in encoder.state_dict().items():
        print(i, k.size(), torch.numel(k))
        cnt += torch.numel(k)
    print(cnt)