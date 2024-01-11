import torch
from Env import Env
#batch_size = 1 for convinience
"""
Sample
-  -  - 
4  1  5
2  3  6
s0 s1 s2
optimal action: (0 -> 2)
"""
samplex = torch.tensor([[[1/2,1/4,0],[1/3,1/1,0],[1/6,1/5,0]]]).to('cpu') 
#n의 역수로 해서 Environment 호환 동시에 Greedy 알고리즘 계산 가능하게 하였음
env = Env('cpu', samplex)
device = 'cpu'
def select_action_GRE_N(x):
    """ Time Complexitiy는 ACO 논문과 다를 수 있습니다.! 제가 비효율적으로 짰을 가능성도 있습니다.
        => 이 알고리즘으로 time을 비교해서는 안됩니다
    """
    def stackof(c): #find stack of block c ACO: s(c)
        #c:Batch
        c_x = c.reshape(-1, 1, 1)
        c_index = torch.nonzero(x == c_x) #Batch X 3 (Batch_i, Stack_i, tier_i)
        return c_index[:,1] # Batch X 1 (Stack_i)
    def CndStacks(c): #Candidate stacks for relocation ACO: R_c
        #c:Batch
        max_stacks = env.max_stacks
        binary_x = torch.where(x > 0., 1, 0)# Block -> 1 Empty -> 0
        stack_len = torch.sum(binary_x, dim=2) #Stack의 Length

        stackofc = stackof(c)#c가 있는 stack
        cnd_stacks = torch.where(stack_len < max_stacks, True, False) # 조건 1: H<MaxH
        cnd_stacks[torch.arange(env.batch), stackofc] = False #조건 2"

    pass
def GRE_N():
    cnt = torch.zeros(env.batch).to(device)
    env.clear()
    while not env.all_empty():
        action = select_action_GRE_N(env.x)
        env.step(action)
        cnt = cnt + (~env.empty).int() #Empty 하지 않을 경우(not ended) + 1
