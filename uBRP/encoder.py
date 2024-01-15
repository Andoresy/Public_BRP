import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary
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
    def forward(self, Q,K,V,mask=None):
        output= self.module(Q,K,V,mask)
        return Q + output

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

    def forward(self, Q, K, V, mask):
        # key, query의 곱을 통해 attention weight를 계산하고 value의 weighted sum인 output을 생성
        # input: Q, K, V, mask (query, key, value, padding 및 시간을 반영하기 위한 mask)
        # output: output, attn_dist (value의 weight sum, attention weight)
        # dim of Q,K,V: batchSize x n_heads x seqLen x d_k(d_v)
        d_k = self.d_k        
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d_k) 
                    # dim of attn_score: batchSize x n_heads x seqLen_Q x seqLen_K
                    #wj) batch matrix multiplication
        print(mask.size())
        print(torch.zeros_like(attn_score).bool().size())
        if mask is None:
            mask = torch.zeros_like(attn_score).bool()
        else:
            mask = mask.unsqueeze(1).repeat(1, Q.size(1), 1, 1)
        attn_score[mask] = -1e9

        attn_dist = F.softmax(attn_score, dim=-1)  # attention distribution
        output = torch.matmul(attn_dist, V)  # dim of output : batchSize x n_heads x seqLen x d_v

        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim, is_encoder=True):
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
        #Noah: 필요하다면 구현해야함
        pass
    def forward(self, Q, K, V, mask):
        batchSize, seqLen_Q, seqLen_K = Q.size(0), Q.size(1), K.size(1) # decoder의 경우 query와 key의 length가 다를 수 있음

        # Query, Key, Value를 (n_heads)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        # dim : batchSize x seqLen x d_model -> batchSize x seqLen x n_heads x d_k
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

        # Linear Projection, Residual sum, and Layer Normalization
        if self.is_encoder:
            output = self.W_O(output)
        
        return output

if __name__ == "__main__":
    mha = MultiHeadAttention(n_heads=8, embed_dim=128, is_encoder=True)
    Smha = SkipConnection(mha)
    norm = Normalization(embed_dim = 128)
    batch,n_nodes,embed_dim = 5, 21, 128
    Q = torch.randn((batch, n_nodes, embed_dim), dtype=torch.float)
    K = Q
    V = Q
    mask = torch.zeros((batch, n_nodes, n_nodes), dtype=torch.bool)
    print("mask: ", mask.shape)
    output = norm(Smha(Q,K,V,mask))
    print("output size:",output.size())
