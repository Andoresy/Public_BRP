import torch
import torch.nn as nn
class Env():
    def __init__(self, device, x):
        super().__init__()
        #x: (batch_size) X (max_stacks) X (max_tiers)
        self.device = device
        self.x = x
        self.batch, self.max_stacks,self.max_tiers=x.size()
        self.target_stack = None
        self.empty = torch.zeros([self.batch], dtype=torch.bool).to(self.device)
        #True -> Empty / False-> not Empty
    def find_target_stack(self):
        #최고 priority Stack 찾기
        mx_val = torch.max(self.x, dim=2)[0].to(self.device)
        self.target_stack = torch.argmax(mx_val, dim=1).to(self.device)
    def _update_empty(self):
        bottom_val = self.x[:,:,0].to(self.device) # 바닥위에 있는 값들 (batch X max_stacks)
        batch_mx = torch.max(bottom_val, dim=1)[0].to(self.device) #Max (batch)
        self.empty = torch.where(batch_mx>0., False, True).to(self.device) #if batch_mx가 0 => Empty
    def clear(self):
        #Retrieve 진행
        self.find_target_stack()
        binary_x = torch.where(self.x > 0., 1, 0).to(self.device) # Block -> 1 Empty -> 0
        stack_len = torch.sum(binary_x, dim=2).to(self.device) #Stack의 Length
        target_stack_len = torch.gather(stack_len, dim=1, index = self.target_stack[:,None].to(self.device)) #target_stack의 location
        stack_mx_index = torch.argmax(self.x, dim=2).to(self.device)
        target_stack_mx_index = torch.gather(stack_mx_index, dim=1, index=self.target_stack[:,None].to(self.device)).to(self.device)
        clear_mask = ((target_stack_len -1) == target_stack_mx_index)
        clear_mask = clear_mask.to(self.device)
        clear_mask = clear_mask & (torch.where(target_stack_len > 0, True, False).to(self.device)) # 완전히 제거된 그룹은 신경쓸 필요 X
        while torch.sum(clear_mask.squeeze(-1))>0:
            batch_mask = clear_mask.repeat_interleave(self.max_stacks * self.max_tiers).to(self.device)
            batch_mask = torch.reshape(batch_mask, (self.batch, self.max_stacks, self.max_tiers)).to(self.device)

            mask = torch.zeros((self.batch, self.max_stacks, self.max_tiers), dtype=torch.bool).to(self.device)
            input_index = (torch.arange(self.batch).to(self.device), self.target_stack, target_stack_len.squeeze(-1).to(self.device) - 1)
            mask = mask.index_put(input_index, torch.tensor(True).to(self.device)).to(self.device)
            # batch_mask에 따라 데이터 그룹에서 최대 값을 지울 수 있음
            mask = mask & batch_mask
            mask = mask.to(self.device)
            self.x = self.x.masked_fill((mask == True).to(self.device), 0.)

            # 동일한 작업을 반복
            self.find_target_stack()
            len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
            stack_len = torch.sum(len_mask, dim=2).to(self.device)
            target_stack_len = torch.gather(stack_len, dim=1, index=self.target_stack[:, None].to(self.device)).to(self.device)
            stack_mx_index = torch.argmax(self.x, dim=2).to(self.device)
            target_stack_mx_index = torch.gather(stack_mx_index, dim=1, index=self.target_stack[:, None].to(self.device)).to(self.device)
            clear_mask = ((target_stack_len - 1) == target_stack_mx_index)
            clear_mask = clear_mask.to(self.device)
            clear_mask = clear_mask & (torch.where(target_stack_len > 0, True, False).to(self.device))

        self._update_empty()
    def step(self, actions):#action Pair로 구성 (source, destination)
        """ action: (batch, (1, 1)) int, range[0,max_stacks)
            no invalid action (Assume)
        """
        len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
        stack_len = torch.sum(len_mask, dim=2)
        source_index = actions[:,0]
        dest_index = actions[:,1]
        source_stack_len = torch.gather(stack_len, dim=1, index=source_index[:,None]).to(self.device)
        dest_stack_len = torch.gather(stack_len, dim=1, index=dest_index[:,None]).to(self.device)
        top_ind = stack_len - 1
        top_ind = torch.where(top_ind >=0, top_ind, 0).to(self.device)
        top_val = torch.gather(self.x, dim=2, index=top_ind[:,:,None]).to(self.device)
        top_val = top_val.squeeze(-1)
        source_top_val = torch.gather(top_val, dim=1, index=source_index[:,None]).to(self.device)
        source_ind = source_stack_len - 1
        source_ind = torch.where(source_ind >=0, source_ind, 0).to(self.device)
        input_index = (torch.arange(self.batch).to(self.device), source_index.to(self.device), source_ind.squeeze(-1).to(self.device))
        self.x = self.x.index_put_(input_index, torch.Tensor([0.]).to(self.device))
        input_index = (torch.arange(self.batch).to(self.device), dest_index.to(self.device), dest_stack_len.squeeze(-1).to(self.device))
        self.x = self.x.index_put_(input_index, source_top_val.squeeze(-1)).to(self.device)
        self.clear()
    def all_empty(self):
        sum = torch.sum(self.empty.type(torch.int))
        if (sum == self.batch):
            return True
        else:
            return False

"""
Sample
-  -  - 
4  1  5
2  3  6
s0 s1 s2
optimal action: (0 -> 2)
"""
samplex = torch.tensor([[[5/6,3/6,0/6],[4/6,6/6,0/6],[1/6,2/6,0/6]],[[5/6,3/6,0/6],[4/6,6/6,0/6],[1/6,2/6,0/6]]]).to('cpu')
sample_action = torch.tensor([[0,1], [0,2]]).to('cpu')
uBRP_Env = Env('cpu', samplex)
uBRP_Env.find_target_stack()
uBRP_Env.clear()
uBRP_Env.step(sample_action)
#print(uBRP_Env.x)