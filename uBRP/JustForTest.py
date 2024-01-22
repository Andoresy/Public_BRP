import torch

original_list = [[[1, 2, 3], [4, 5, 6], [7, 8, 0]], [[7, 8, 9], [10, 11, 22], [4, 5, 6]]]

# 리스트를 PyTorch 텐서로 변환
tensor = torch.tensor(original_list)

# 텐서의 각 행을 오른쪽으로 한 칸씩 이동
shifted_tensor = torch.cat([tensor[:, -1:], tensor[:, :-1]], dim=1)

# 이동된 텐서를 리스트로 변환
shifted_list = shifted_tensor.tolist()

print(shifted_list)