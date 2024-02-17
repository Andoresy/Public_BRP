import copy
import torch
import torch.nn as nn

max_num = 5
type_of_Size = sorted([(i,j) for i in range(3,max_num+1) for j in range(max(i-1, 3), max_num+1)], key = lambda x: x[0]*x[1]) #Should be Tested
print(torch.randint(low=0, high=1, size=(1,)).repeat(10))