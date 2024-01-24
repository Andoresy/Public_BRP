from baseline import load_model
from data import generate_data
from GreedyACO import GRE_rBRP, GRE_uBRP
from data import data_from_caserta, data_from_caserta_for_greedy
import torch
import gc
if __name__ == '__main__':
    device = 'cuda:0'
    H,W = 5,4 # ACO 논문 기준 H X W = T X S
    H_plus = 2
    N = H*W
    #for i in [2,3]:
    for i in range(0, 10):
        path = f"./Train/Exp81/epoch{i}.pt"
        model = load_model(device='cuda:0', path=path,n_encode_layers=4, embed_dim=128, n_containers=N, max_stacks=W, max_tiers=H+H_plus)
        data_caserta = data_from_caserta(f'data{H}-{W}-.*', H_plus).to(device)
        data_greedy = data_from_caserta_for_greedy(f'data{H}-{W}-.*', H_plus).to(device)
#        print(data_caserta.size())
#        print(data_greedy.size())
        return_pi = False
        output = model(data_caserta, decode_type='greedy', return_pi=return_pi)
        output_ = output[0]
        print(f"Greedy Mean Locations for {i}th epoch:",output[0].mean())  # cost: (batch)
        is_toobig = torch.torch.where(output_ > 50, True, False)
        is_toobig = torch.nonzero(is_toobig).squeeze()
        #is_toobig_sam = is_toobig[0]
        #print(output_[is_toobig])
        #model(data[is_toobig_sam:is_toobig_sam+1],decode_type='greedy', return_pi=True)
#    print(output[1])  # ll: (batch)
    device = 'cuda:0'


    #---greedy
    """
    rBRP_cnt = 0
    uBRP_cnt = 0
    rBRP_cnt_ar = []
    uBRP_cnt_ar = []
    cnt = 0
    for d in data_greedy:
        d1 = d.clone().unsqueeze(0).to(device)
        #print(d1)
        d2 = d.clone().unsqueeze(0).to(device)
        cnt+=1
        t1 = GRE_rBRP(d1)
        rBRP_cnt += t1
        rBRP_cnt_ar.append(t1)
        t2 = GRE_uBRP(d2)
        uBRP_cnt_ar.append(t2)
        uBRP_cnt += t2
        gc.collect()
    print(f"H x W (T x S in ACO) : {H} x {W}")
    print(f"Test_cnt: {cnt}개")
    print(f"Avg rBRP_cnt: {rBRP_cnt/cnt}")
    print(f"Avg uBRP_cnt: {uBRP_cnt/cnt}")
    for i in range(cnt):
        print(f"rBRPGR({i}) : {rBRP_cnt_ar[i]} uBRPGR({i}) : {uBRP_cnt_ar[i]} Model({i}): {output_[i]}")  
        """