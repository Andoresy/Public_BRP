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


matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 변경할 인덱스들 정의
rows_to_change = [0, 2]
columns_to_change = [1, 2]
new_values = [10, 20]

# 여러 인덱스의 값을 동시에 바꾸기
matrix[rows_to_change, columns_to_change] = torch.tensor(new_values)

print(matrix)