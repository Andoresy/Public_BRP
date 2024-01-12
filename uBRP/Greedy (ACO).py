import torch
from Env import Env
""" batch_size = 1 for computational convinience
Sample 1
-  -  7 
4  3  5
2  1  6
s0 s1 s2
optimal action in uBRP: (1->0) (2->1) (0->2) (0->1)
optimal action in rBRP: (1->0) (0->1) (0->1) (1->0) (2->0)
cnt = 4
"""
samplex = torch.tensor([[[1/2,1/4,0],[1/1,1/3,0],[1/6,1/5,1/7]]]).to('cpu') 
#n의 역수로 해서 Environment 호환 동시에 Greedy 알고리즘 계산 가능하게 하였음
global env
env = Env('cpu', samplex)
device = 'cpu'
N=7
def reset():
    global env
    samplex = torch.tensor([[[1/2,1/4,0],[1/1,1/3,0],[1/6,1/5,1/7]]]).to('cpu') 
    env = Env('cpu', samplex)
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
        c_index = torch.nonzero(x == c)
        return c_index[0][0] # (Stack_i)
    def CndStacks(c): #Candidate stacks for relocation ACO: R_c (1)
        #c:Batch
        stackofc = stackof(c)#c가 있는 stack
        cnd_stacks = torch.where(stack_len < max_tiers, True, False) # 조건 1: H<MaxH
        cnd_stacks[stackofc] = False #조건 2: S != s(c)
        return torch.squeeze(cnd_stacks.nonzero(), -1)
    def dd(S): #due_date(Stack_index) min of stack ACO: (2)
        if stack_len[S]==0:
            return N+1
        return 1/torch.max(x[S])
    def dif(c, S): # ACO: (3,4)
        c = 1/c
        d = dd(S)
        if d>c:
            return d-c
        else:
            return 2*N+1-d
    def well_Located(c):
        S = stackof(c)
        return dd(S) >= 1/c
    def MinMax_rBRP(c): # ACO: (5)
        Candidate_Stacks = CndStacks(c)
        best_cs,best_dif = None, 999 #Best_candidateStack
        for c_s in Candidate_Stacks:
            t_dif = dif(c, c_s)
            #print("DEB",c,c_s, t_dif)
            if t_dif < best_dif:
                best_cs = c_s
                best_dif = t_dif
        return best_cs
    def well_locate_CndStacks(c): # ACO: (9)
        stackofc = stackof(c)#c가 있는 stack
        cnd_stacks = torch.where(stack_len < max_tiers, True, False) # 조건 1: H<MaxH
        cnd_stacks[stackofc] = False #조건 2: S != s(c)
        for S in range(max_stacks):
            if(dd(S) <= 1/c):
                cnd_stacks[S] = False
        return torch.squeeze(cnd_stacks.nonzero(), -1)
    def tn(): # Tops that are not well located and not on Target ACO: (10)
        top_blocks = tops()
        temp_tops = []
        for b in top_blocks:
            if (not well_Located(b)) and stackof(b) != target_stack:
                temp_tops.append(b)
        return torch.tensor(temp_tops)
    def Or(): # ACO: (12)
        Tn = tn()
        Or_ = []
        for c in Tn:
            Wc = well_locate_CndStacks(c)
            for w in Wc:
                Or_.append([stackof(c),Wc])
        return torch.tensor(Or_, dtype=torch.long)
    def MinMax_uBRP(): #ACO: (14)
        T = torch.tensor([[target_stack, MinMax_rBRP(top(target_stack))]])# ACO: (11) 조금 변형됨 rBRP 중 best 가지고옴
        Cr = torch.cat([T, Or()])
        best_action,best_dif = torch.tensor([[0,0]]), 999 #Best_candidateStack & initialize
        for action in Cr: #sourceStack, destStack
            sS,dS = action
            t_dif = dif(top(sS), dS)
            #print("DEB",c,c_s, t_dif)
            if t_dif < best_dif:
                best_action[0] = action
                best_dif = t_dif
        return best_action
    if algo_type == "Greedy_rBRP":
        return torch.tensor([[target_stack, MinMax_rBRP(top(target_stack))]])
    elif algo_type == "Greedy_uBRP":
        return MinMax_uBRP()

    return None
def GRE_rBRP():
    cnt = 0
    reset()
    env.clear()
    while not env.all_empty():
        action = select_action_GREEDY(env.x[0], "Greedy_rBRP") #Batch = 1 (문제 하나)
        print(action, end=" ")
        env.step(action)
        cnt = cnt + 1
    print("Greedy in rBRP steps: ", cnt)
def GRE_uBRP():
    cnt = 0
    reset()
    env.clear()
    while not env.all_empty():
        
        action = select_action_GREEDY(env.x[0], "Greedy_uBRP") #Batch = 1 (문제 하나)
        print(action, end=" ")
        env.step(action)
        cnt = cnt + 1
    print("Greedy in uBRP steps: ", cnt)
GRE_rBRP()
GRE_uBRP()