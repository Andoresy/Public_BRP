import torch
import torch.nn as nn
from data import generate_data
class Env():
    def __init__(self, device, x, embed_dim=128):
        super().__init__()
        #x: (batch_size) X (max_stacks) X (max_tiers)
        self.device = device
        self.x = x
        self.embed_dim = embed_dim
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
        target_stack_mx_index = torch.gather(stack_mx_index, dim=1, index=self.target_stack[:,None].to(self.device))
        clear_mask = ((target_stack_len -1) == target_stack_mx_index)
        clear_mask = clear_mask
        clear_mask = clear_mask & (torch.where(target_stack_len > 0, True, False)) # 완전히 제거된 그룹은 신경쓸 필요 X
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
    def _get_step(self, next_node):
        """ next_node : (batch, 1) int, range[0, max_stacks)

            mask(batch,max_stacks,1) 1表示那一列不可选，0表示可选
            context: (batch, 1, embed_dim)
        """

        # len_mask(batch,max_stacks,max_tiers)表示把len的值都弄出来变成1和0
        len_mask = torch.where(self.x > 0., 1, 0).to(self.device)
        # stack_len(batch,max_stacks)表示每一列最高是多少
        stack_len = torch.sum(len_mask, dim=2)
        # self.target_stack[:,None](batch,1)
        # target_stack_len(batch,1)[i][0]=stack_len[i][target_stack[i][0]] 相当于把目标列的长度都拿出来了

        target_stack_len = torch.gather(stack_len, dim=1, index=self.target_stack[:, None]).to(self.device)

        #next_stack_len(batch,1)[i][0]=stack_len[i][next_node[i][0]]
        next_stack_len=torch.gather(stack_len,dim=1,index=next_node).to(self.device)

        #  top_val(batch,max_stacks,1)=x(batch,max_stacks,stack_len(i,j,1))
        #top_ind(batch,max_stacks)-1 ,再用where把空的边0
        top_ind=stack_len-1
        top_ind=torch.where(top_ind>=0,top_ind,0).to(self.device)
        #top_val(batch,max_stacks,1)=x(batch,max_stacks,top_ind(i,j,1))
        top_val=torch.gather(self.x,dim=2,index=top_ind[:,:,None]).to(self.device)
        #top_val(batch,max_stacks)
        top_val=top_val.squeeze(-1)
        #target_top_val(batch,1)=top_val(batch,self.target_stack[batch,1])
        target_top_val=torch.gather(top_val,dim=1,index=self.target_stack[:,None]).to(self.device)

        #target_ind(batch,1)
        target_ind=target_stack_len-1
        target_ind=torch.where(target_ind>=0,target_ind,0).to(self.device)
        input_index=(torch.arange(self.batch).to(self.device),self.target_stack.to(self.device),target_ind.squeeze(-1).to(self.device))
        self.x=self.x.index_put_(input_index,torch.Tensor([0.]).to(self.device))


        input_index=(torch.arange(self.batch).to(self.device),next_node.squeeze(-1).to(self.device),next_stack_len.squeeze(-1).to(self.device))
        self.x=self.x.index_put_(input_index,target_top_val.squeeze(-1)).to(self.device)
        self.clear()
    def step(self, actions):#action Pair로 구성 (source, destination)
        """ action: (batch, 2) int, range[0,max_stacks)
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
#        print("input_index:",input_index)
        self.x = self.x.index_put_(input_index, source_top_val.squeeze(-1)).to(self.device)
        self.clear()
    def all_empty(self):
        sum = torch.sum(self.empty.type(torch.int))
        if (sum == self.batch):
            return True
        else:
            return False
    """ 밑에서부턴 https://github.com/binarycopycode/CRP_DAM/blob/main 코드부분과 일치합니다.(코드 이해 필요)
        uBRP에서의 수정 필요
    """
    def create_mask_uBRP(self):
        top_val=self.x[:,:,-1].to(self.device)
        bottom_val = self.x[:,:,0].to(self.device)
        mask_top=torch.where(top_val>0,True,False).to(self.device).bool()
        mask_top = mask_top.repeat(1, self.max_stacks).view(self.batch, self.max_stacks, self.max_stacks).to(self.device)
        mask_bottom=torch.where(bottom_val==0.,True,False).to(self.device).bool()
        mask_bottom = mask_bottom.view(self.batch, self.max_stacks, 1).repeat(1, 1, self.max_stacks).to(self.device)

        mask = torch.logical_or(mask_top, mask_bottom).to(self.device)
        diagonal_size = self.max_stacks
        d_tensor = torch.zeros((self.batch, diagonal_size, 2), dtype=torch.long).to(self.device)
        # 대각선 값 설정
        d_tensor[:,:,0] = torch.arange(diagonal_size).to(self.device)
        d_tensor[:,:,1] = torch.arange(diagonal_size).to(self.device)
        mask.scatter_(2, d_tensor, 1)
        return mask.view(self.batch, self.max_stacks*self.max_stacks)[:,:,None].to(self.device)
    def create_context_uBRP(self):
        pass
    def _create_t1(self):
        self.find_target_stack()
        #mask(batch,max_stacks,1) 1表示那一列不可选，0表示可选
        mask_t1 = self.create_mask_t1()
        #step_context = target_stack_embedding(batch, 1, embed_dim)
        step_context_t1= self.create_context_t1()
        return mask_t1, step_context_t1

    def create_mask_t1(self):

        top_val=self.x[:,:,-1]
        mask=torch.where(top_val>0,True,False).to(self.device)
        mask = mask.bool()

        #当前target_stack也不能走,这几句相当于mask[i][target_stack[i]]=True
        a=self.target_stack.clone().to(self.device)
        index = (torch.arange(self.batch).to(self.device), a.squeeze())
        mask=mask.index_put(index,torch.BoolTensor([True] ).to(self.device))

        #mask(batch,max_stacks,1) 1表示那一列不可选，0表示可选
        return mask[:,:,None].to(self.device)

    def create_context_t1(self):
        


        # node_embeddings(batch,max_stacks,embed_dim) 然后把target_stack变成(batch,1,1)后把最后一维循环embed_dim变成(batch,1,128) 然后使用gather dim=1 就相当于
        # target_stack_embedding(batch,1,embed_dim)=node_embeddings(i,idx[i][j][k],k)，就是把目标列的embeding全部拿出来了

        target_stack_embedding = torch.gather(input=self.node_embeddings, dim=1,
                                       index=self.target_stack[:, None, None].repeat(1, 1, self.embed_dim))
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4


        return target_stack_embedding

    def get_log_likelihood(self, _log_p, pi):
        """ _log_p: (batch, decode_step, n_nodes)
            pi: (batch, decode_step), predicted tour
        """
        # log_p(batch,decode_step,1) 相当于把pi这个路径的所有log_p值给弄出来了
        # 相当于log_p(i,j,k=0)=_log_p(i,j,pi[i][j][0])
        log_p = torch.gather(input=_log_p, dim=2, index=pi[:, :, None])
        # 先把最后1维的1给消掉，然后再加起来
        # 由于是log_softmax,所以概率本来要用乘法，取了log就可以加法了
        return torch.sum(log_p.squeeze(-1), 1)

if __name__ == "__main__":
    data = generate_data('cuda:0')
    env = Env('cuda:0', data[:2])
    env.node_embeddings = torch.randn((2,4,2))
    env.embed_dim = 2
    env.clear()
    print(env.x)

