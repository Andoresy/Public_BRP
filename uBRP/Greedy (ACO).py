import torch
from Env import Env
""" batch_size = 1 for computational convinience
Sample
-  -  - 
4  3  5
2  1  6
s0 s1 s2
optimal action: (1 -> 2) , (0 -> 1)
cnt = 2
"""
samplex = torch.tensor([[[1/2,1/4,0],[1/1,1/3,0],[1/6,1/5,0]]]).to('cpu') 
#n의 역수로 해서 Environment 호환 동시에 Greedy 알고리즘 계산 가능하게 하였음
env = Env('cpu', samplex)
device = 'cpu'
N=6
def select_action_GREEDY(x, algo_type="Greedy_rBRP"): 
    """ Time Complexitiy는 ACO 논문과 다를 수 있습니다.! 제가 비효율적으로 짰을 가능성도 있습니다.
        => 이 알고리즘으로 time을 비교해서는 안됩니다
        algo_type: Greedy_rBRP, Greedy_uBRP, Greedy_uBRP_Extend
    """
    
    binary_x = torch.where(x > 0., 1, 0)# Block -> 1 Empty -> 0
    stack_len = torch.sum(binary_x, dim=1) #Stack의 Length
    max_tiers = env.max_tiers
    max_stacks = env.max_stacks
    target_stack = env.target_stack[0]
    def top(s): #top of stack
        return x[s,stack_len[s]-1]
    def tops():
        s = torch.arange(max_stacks)
        tps = torch.tensor([])
        for i in s:
            if stack_len[i] > 0:
                tps = torch.cat([tps, torch.tensor([x[i,stack_len[i]-1]])])
        return tps
    def stackof(c): #find stack of block c ACO: s(c)
        #c:Batch
        c_index = torch.nonzero(x == c) #Batch X 3 (Batch_i, Stack_i, tier_i)
        return c_index[0][0] # (Stack_i)
    def CndStacks(c): #Candidate stacks for relocation ACO: R_c
        #c:Batch
        stackofc = stackof(c)#c가 있는 stack
        cnd_stacks = torch.where(stack_len < max_tiers, True, False) # 조건 1: H<MaxH
        cnd_stacks[stackofc] = False #조건 2: S != s(c)
        return torch.squeeze(cnd_stacks.nonzero(), -1)
    def dd(S): #due_date(Stack_index) min of stack
        if stack_len[S]==0:
            return N+1
        return 1/torch.max(x[S])
    def dif(c, S):
        c = 1/c
        d = dd(S)
        if d>c:
            return d-c
        else:
            return 2*N+1-d
    def MinMax_rBRP(c):
        Candidate_Stacks = CndStacks(c)
        best_cs,best_dif = None, 999 #Best_candidateStack
        for c_s in Candidate_Stacks:
            t_dif = dif(c, c_s)
            #print("DEB",c,c_s, t_dif)
            if t_dif < best_dif:
                best_cs = c_s
                best_dif = t_dif
        return best_cs
    def well_locate_CndStacks(c):
        stackofc = stackof(c)#c가 있는 stack
        cnd_stacks = torch.where(stack_len < max_tiers, True, False) # 조건 1: H<MaxH
        cnd_stacks[stackofc] = False #조건 2: S != s(c)
        for S, cnd in range(max_stacks):
            if(dd(S) <= 1/c):
                cnd_stacks[S] = True
    def MinMax_uBRP():
    
    print(tops())
    if algo_type == "Greedy_rBRP":
        return torch.tensor([[target_stack, MinMax_rBRP(top(target_stack))]])
    elif algo_type == "Greedy_uBRP":
        return None

    return None
def GRE_rBRP():
    cnt = 0
    env.clear()
    while not env.all_empty():
        action = select_action_GREEDY(env.x[0], "Greedy_rBRP") #Batch = 1 (문제 하나)
        env.step(action)
        cnt = cnt + 1
    print("Greedy in rBRP steps: ", cnt)
GRE_rBRP()