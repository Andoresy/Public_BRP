import torch
import torch.nn as nn
import numpy as np
from data import generate_data
from Env_V2 import Env
from scipy.stats import truncexpon
"""
original_list = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 0]], [[-1, 2, 3], [4, 5, 6], [1, 2, 0.]]])

# 리스트를 PyTorch 텐서로 변환
batch,max_stacks,max_tiers,embed_dim=2,3,3,32
original_list = original_list.view(batch, max_stacks, max_tiers, 1)
linear = nn.Linear(1, embed_dim, bias=True)
x = linear(original_list)
lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=1, bidirectional=False, batch_first=True)
x = x.view(batch*max_stacks, max_tiers, embed_dim)
outputs, (hidden_state, cell_state) = lstm(x)
print(hidden_state.view(batch, max_stacks, embed_dim).size())
"""
states = [(i,j) for i in range(3,11) for j in range(i-1, 11)]
# for repeatability:
import numpy as np
np.random.seed(0)
print(len(states))
from scipy.stats import poisson, uniform
from scipy import stats
sample_size = 20
maxval = 44
mu = 0.01
lower, upper, scale = 0, 43, .5

for i in range(400):
    scale = scale*1.03
    X = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
    data = X.rvs(1000)
    print(i, scale,  np.rint(data).mean())