import copy
import torch
import torch.nn as nn

my_tensor = torch.tensor([[2, 0, 1, 4],[2,1,0,3]], dtype=torch.float32)
t = torch.tensor([1, 2])

height =torch.tensor([1, 2, 5])
height = 10 - height

h_t = (torch.arange(10, 0, -1).view(1,1,10).repeat(1,3,1) - height.view(1,3,1).repeat(1,1,10))
print(torch.where(h_t < 0, 0, h_t))