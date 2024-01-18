import torch

# A와 B 정의
A = torch.tensor([True, False, True, False])
B = torch.tensor([[3, 2, 1], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# A의 True에 해당하는 B의 요소를 0으로 만듦
B[A] = torch.tensor([0, 0, 0])

print("변경된 B:\n", B)