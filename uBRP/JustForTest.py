import torch
import torch.nn as nn
# A와 B 정의
A = torch.randn([1, 2, 2])
conv1 = nn.Conv1d(3, 3, 3, 1, 1)
print(A)
print(A.repeat(3, 1, 1))
#print(A)
#print(torch.cat([nn.Conv1d(3, 3, 3, 1, 1)(A), A], dim=-1))
    