import torch
import numpy as np
from data import generate_data
from Env_V2 import Env
original_list = [[[1, 2, 3], [4, 5, 6], [7, 8, 0]], [[7, 8, 9], [10, 11, 22], [4, 5, 6]]]

# 리스트를 PyTorch 텐서로 변환
batch,max_stacks,max_tiers = 2, 3, 4
n_containers = max_stacks *(max_tiers-2)
data = generate_data('cuda:0', batch, n_containers, max_stacks, max_tiers)
binary_x = torch.where(data > 0., 1, 0).to('cuda:0') # Block -> 1 Empty -> 0
stack_len = torch.sum(binary_x, dim=2).to('cuda:0') #Stack의 Length
block_nums = torch.sum(stack_len, dim=1).to('cuda:0')

env = Env('cuda:0', data)
print(data)
env.clear()
print(env.x)
binary_x = torch.where(env.x > 0., 1, 0).to('cuda:0') # Block -> 1 Empty -> 0
stack_len = torch.sum(binary_x, dim=2).to('cuda:0') #Stack의 Length
new_block_nums = torch.sum(stack_len, dim=1).to('cuda:0')
print(block_nums)
print(new_block_nums)
new_ratio = (block_nums + 1)/(new_block_nums+1)
print(new_ratio)
print(torch.mul(env.x.view(batch, max_stacks*max_tiers), new_ratio.unsqueeze(1)).view(batch, max_stacks, max_tiers))
