import torch
import numpy as np
original_list = [[[1, 2, 3], [4, 5, 6], [7, 8, 0]], [[7, 8, 9], [10, 11, 22], [4, 5, 6]]]

# 리스트를 PyTorch 텐서로 변환
tensor = torch.tensor(original_list)

# 텐서의 각 행을 오른쪽으로 한 칸씩 이동
shifted_tensor = torch.cat([tensor[:, -1:], tensor[:, :-1]], dim=1)

# 이동된 텐서를 리스트로 변환
shifted_list = shifted_tensor.tolist()

#print(shifted_list)
graph_embedding = torch.randn([5, 128]).view(5, 1, 128)
node_embedding = torch.randn([5,16,200])
extd_grpah_embedding = graph_embedding.repeat([1, 16, 1])

n_containers = 6
max_stacks = 3
max_tiers = 4
device = 'cpu'
plus_tiers = 3
plus_stacks = 2
per = np.arange(0, n_containers, 1)
np.random.shuffle(per)
per=torch.FloatTensor((per+1)/(n_containers+1.0))
data=torch.reshape(per,(max_stacks,max_tiers-2)).to(device)
data = torch.cat([torch.zeros(plus_stacks, max_tiers-2),data], dim=0)
data = data[torch.randperm(data.size()[0])]
add_empty=torch.zeros((max_stacks+plus_stacks,plus_tiers),dtype=float).to(device)
print(torch.cat( (data,add_empty) ,dim=1).to(device))