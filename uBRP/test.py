from baseline import load_model
from data import generate_data
from GreedyACO import GRE_rBRP, GRE_uBRP
import torch
if __name__ == '__main__':

    model = load_model('cpu', path="./epoch5.pt", embed_dim=128, n_containers=8, max_stacks=4, max_tiers=4)
    data = generate_data('cpu', n_samples=400)
    return_pi = False
    output = model(data, decode_type='sampling', return_pi=return_pi)
    print("Mean Locations:",output[0].mean())  # cost: (batch)
#    print(output[1])  # ll: (batch)




    #---greedy
    H,W = 2,4 # ACO 논문 기준 H X W = T X S
    rBRP_cnt = 0
    uBRP_cnt = 0
    cnt = 0
    for d in data:
        d1 = d.clone().unsqueeze(0)
        #print(d1)
        d2 = d.clone().unsqueeze(0)
        cnt+=1
        rBRP_cnt += GRE_rBRP(d1)
        uBRP_cnt += GRE_uBRP(d2)
    print(f"H x W (T x S in ACO) : {H} x {W}")
    print(f"Test_cnt: {cnt}개")
    print(f"Avg rBRP_cnt: {rBRP_cnt/cnt}")
    print(f"Avg uBRP_cnt: {uBRP_cnt/cnt}")
