import torch
import torch.nn as nn

from data import generate_data
from decoder_uBRP_LSTM import Decoder_uBRP

class AttentionModel_LSTM(nn.Module):

    def __init__(self,device, embed_dim=128, n_encode_layers=3, n_heads=8,
                 tanh_clipping=10., ff_hidden=512,n_containers=8, max_stacks=4,max_tiers=4):
        super().__init__()


        self.Decoder = Decoder_uBRP(device=device,n_encode_layers=n_encode_layers,embed_dim=embed_dim, n_heads=n_heads, clip=tanh_clipping,ff_hidden=ff_hidden,
                                   n_containers=n_containers,max_stacks=max_stacks,max_tiers=max_tiers)
        self.n_containers=n_containers
        self.max_stacks=max_stacks
        self.max_tiers=max_tiers
        self.device=device

    # cost length ll:log_softmax  sum of probability  pi:predicted tour
    def forward(self, x, return_pi=False, decode_type='greedy'):
        decoder_output = self.Decoder(x,self.n_containers, return_pi=return_pi, decode_type=decode_type)
        if return_pi:
            cost, ll, L = decoder_output
            return cost, ll, L
        cost, ll, L = decoder_output
        return cost, ll, L


if __name__ == '__main__':

    model = AttentionModel('cpu')
    model.train()
    data = generate_data('cpu', n_samples=5)
    return_pi = False
    output = model(data, decode_type='sampling', return_pi=return_pi)
    if return_pi:
        cost, ll, pi = output
        print('\ncost: ', cost.size(), cost)
        print('\nll: ', ll.size(), ll)
        print('\npi: ', pi.size(), pi)
    else:
        print(output[0])  # cost: (batch)
        print(output[1])  # ll: (batch)

    cnt = 0
    for i, k in model.state_dict().items():
        print(i, k.size(), torch.numel(k))
        cnt += torch.numel(k)
    print('total parameters:', cnt)

# output[1].mean().backward()
# print(model.Decoder.Wout.weight.grad)
# print(model.Encoder.init_W_depot.weight.grad)