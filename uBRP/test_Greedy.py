from baseline import load_model
from data import generate_data
from data import data_from_caserta, data_from_caserta_for_greedy
import torch
import gc
if __name__ == '__main__':
    device = 'cuda:0'
    HWS = [(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(4,4),(4,5),(4,6),(4,7),(5,4),(5,5),(5,6),(5,7), (5,8), (5,9), (5,10),(6,6)]
    #HWS = [(3,3), (3,4), (3,5), (3,6),(4,4),(5,6), (5,7), (3,8)]
    for H,W in HWS:
        H_plus = 2
        N = H*W
        Exp_num= 121
        epoch_num = 337
        embed_dim = 64
        data_caserta = data_from_caserta(f'data{H}-{W}-.*', H_plus).to(device)
        data_greedy = data_from_caserta_for_greedy(f'data{H}-{W}-.*', H_plus).to(device)
        for i in range(epoch_num,epoch_num+1):
    #        print(data_caserta.size())
    #        print(data_greedy.size())
            path = f"./train/Exp{Exp_num}/epoch{i}.pt"
            model = load_model(device='cuda:0', path=path,n_encode_layers=3, embed_dim=embed_dim, n_containers=N, max_stacks=W, max_tiers=H+H_plus, is_Test = True).to(device)
            model.eval()
            return_pi = False
            output = model(data_caserta, decode_type='greedy', return_pi=return_pi)
            output_ = output[2] #Length
            print(f"{H}X{W} Greedy Mean Locations for {i}th epoch:",output[2].mean())  # cost: (batch)
            print(output_)
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