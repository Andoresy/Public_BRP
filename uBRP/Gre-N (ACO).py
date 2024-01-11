import torch
from Env import Env
#batch_size = 1 for convinience
samplex = torch.tensor([[[5/6,3/6,0/6],[4/6,6/6,0/6],[1/6,2/6,0/6]]]).to('cpu')
env = Env('cpu', samplex)
device = 'cpu'
def select_action_GRE_N(x):
    pass
def GRE_N():
    cnt = torch.zeros(env.batch).to(device)
    while not env.all_empty():
        action = select_action_GRE_N(env.x)
        env.step(action)
        cnt = cnt + (~env.empty).int() #Empty 하지 않을 경우(not ended) + 1
        