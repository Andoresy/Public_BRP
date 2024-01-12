import torch

original_tensor = torch.tensor([[[1/2,1/4,0],[1/3,1/1,0],[1/6,1/5,0]]])

# 원하는 값들을 리스트로 지정
c = torch.tensor([1,3])
target_values = c.reshape(-1, 1, 1)

# 조건을 설정합니다.
c_index = torch.nonzero(original_tensor == target_values)
#print(c_index[:,1])
x = original_tensor
binary_x = torch.where(x > 0., 1, 0)# Block -> 1 Empty -> 0
stack_len = torch.sum(binary_x, dim=2) #Stack의 Length
stack_len = torch.where(stack_len < 3, True, False)
#print(stack_len)


x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
def stackof(c): #find stack of block c ACO: s(c)
    #c:Batch
    c_index = torch.nonzero(x == c) #Batch X 3 (Batch_i, Stack_i, tier_i)
    return c_index[0][0] # (Stack_i)
#print(matrix)
#print(torch.squeeze(torch.gather(maxs, 1, torch.tensor([[0],[2]])), -1))
max_stacks = 3
binary_x = torch.where(x > 0., 1, 0)# Block -> 1 Empty -> 0
stack_len = torch.sum(binary_x, dim=1) #Stack의 Length
stackofc = stackof(1)#c가 있는 stack
cnd_stacks = torch.where(stack_len < max_stacks, True, False) # 조건 1: H<MaxH
cnd_stacks[stackofc] = False #조건 2: S != s(c)

#print(cnd_stacks)

print(torch.cat([torch.tensor([1]),torch.tensor([2])]))