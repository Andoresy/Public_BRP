import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from datetime import datetime
import os
from model_LSTM import AttentionModel_LSTM
from baseline import RolloutBaseline, load_model
from data import generate_data, Generator, MultipleGenerator

def train(log_path = None, dict_file = None):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device!='cpu':
        torch.cuda.set_device(device)
    n_encode_layers = dict_file["n_encode_layers"] # 4
    N_samplings = dict_file["N_samplings"] 
    epochs = dict_file["epochs"] 
    batch = dict_file["batch"] 
    batch_num = dict_file["batch_num"] 
    batch_verbose = dict_file["batch_verbose"] 
    max_stacks = dict_file["max_stacks"] 
    max_tiers = dict_file["max_tiers"]
    baseline_type = dict_file["baseline_type"]
    plus_tiers = dict_file["plus_tiers"]
    lr = dict_file["lr"]
    beta = dict_file["beta"]
    embed_dim = dict_file["embed_dim"]
    warmuplr = dict_file["warmuplr"]
    n_containers = max_stacks*(max_tiers-2)
    model_save_path = log_path
    log_path = log_path + f'/{max_stacks}X{max_tiers-2}Problem_NoAug_x{N_samplings}_Linearx2Init_{n_encode_layers}_layers_0_epoch{epochs}.txt'

# 파일이 이미 존재하는지 확인
    
    # open w就是覆盖，a就是在后面加 append
    with open(log_path, 'w') as f:
        f.write(datetime.now().strftime('%y%m%d_%H_%M'))
    with open(log_path, 'a') as f:
        f.write('\n start training \n')
        f.write(dict_file.__str__())
    
    model = AttentionModel_LSTM(device=device, n_encode_layers=n_encode_layers, embed_dim=embed_dim, max_stacks = max_stacks, max_tiers = max_tiers+plus_tiers-2, n_containers = n_containers)
    #model = AttentionModel(device=device, n_encode_layers=n_encode_layers, embed_dim=128, max_stacks = max_stacks, max_tiers = max_tiers+plus_tiers-2, n_containers = n_containers)
    path = "./Train/Exp21/epoch71.pt" #from previous version
    #model = load_model(device='cuda:0', path=path,n_encode_layers=4, embed_dim=embed_dim, n_containers=n_containers, max_stacks=max_stacks, max_tiers=max_tiers+plus_tiers-2)
    model=model.to(device)
    model.train()

    baseline = RolloutBaseline(model, task=None,  device=device,weight_dir = None, log_path=log_path, max_stacks = 5, max_tiers = 7, plus_tiers = plus_tiers,n_containers = 25)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: .99 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
    #bs batch steps number of samples = batch * batch_steps
    def rein_loss(model, inputs, bs, t, device):
        with torch.no_grad():
            if baseline_type == 'greedy':
                b, bl = baseline.model(inputs, decode_type = 'greedy')
            elif baseline_type == 'sampling':
                bLs = torch.zeros([batch]).to(device)
                for i in range(N_samplings):
                    bL, bll = baseline.model(inputs, decode_type='sampling')
                    bLs = bLs + bL
                b = bLs/N_samplings
            elif baseline_type == 'augmented_sampling':
                bLs = torch.zeros([batch]).to(device)
                shifted_input = inputs
                for i in range(N_samplings):
                    shifted_input = torch.cat([shifted_input[:, -1:], shifted_input[:, :-1]], dim=1)
                    bL, bll = baseline.model(shifted_input, decode_type='sampling')
                    bLs = bLs + bL
                b = bLs/N_samplings
            elif baseline_type == 'greedy+augmented_sampling':
                bLs = torch.zeros([batch]).to(device)
                shifted_input = inputs
                bG, bgl = baseline.model(inputs, decode_type = 'greedy')
                for i in range(N_samplings):
                    shifted_input = torch.cat([shifted_input[:, -1:], shifted_input[:, :-1]], dim=1)
                    bL, bll = baseline.model(shifted_input, decode_type='sampling')
                    bLs = bLs + bL
                bS = bLs/N_samplings
                b = bG*beta + bS*(1-beta)
            elif baseline_type == 'greedy+new_sampling':
                b, bl = baseline.model(inputs, decode_type = 'greedy')
                b_news = [b.unsqueeze(0)]
                for i in range(N_samplings):
                    b_new, b_newl = baseline.model(inputs, decode_type = 'new_sampling')
                    b_news.append(b_new.unsqueeze(0))
                #print(b_news)
                best_from_new = torch.cat(b_news).min(dim=0)[0]
                b = best_from_new

        model.train()
        L, ll = model(inputs, decode_type='sampling')
        #b = bs[t] if bs is not None else baseline.eval(inputs, L)
        return ((L - b) * ll).mean(), L.mean()
        #return ((L-bL)*ll).mean(), L.mean()

    tt1 = time()


    t1=time()
    for epoch in range(epochs):
        ave_loss, ave_L = 0., 0.

        datat1=time()
        n_containers = max_stacks * (max_tiers-2)
        datasets=MultipleGenerator(device, batch=batch, n_samples=batch*batch_num, epoch=epoch).get_dataset()
        datat2=time()
        print('data_gen: %dmin%dsec' % ((datat2 - datat1) // 60, (datat2 - datat1) % 60))

        #bs=baseline.eval_all(dataset)
        #bs = bs.view(-1, batch) if bs is not None else None  # bs: (cfg.batch_steps, cfg.batch) or None

        model.train()
        dataloaders = [DataLoader(dataset, batch_size=batch, shuffle=True) for dataset in datasets]
        for t, dataloader in enumerate(dataloaders):
            for inputs in dataloader:
                #print(inputs.size())
                loss,L_mean=rein_loss(model,inputs,None,t,device)
                optimizer.zero_grad()
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                optimizer.step()

                ave_loss += loss.item()
                ave_L += L_mean.item()

            if t % (batch_verbose) == 0:
                t2 = time()
                print('Epoch %d (batch = %d): Loss: %1.3f avg_L: %1.3f, batch_Loss: %1.3f batch_L: %1.3f %dmin%dsec' % (
                    epoch, t, ave_loss / (t + 1), ave_L / (t + 1), loss.item(), L_mean.item(), (t2 - t1) // 60, (t2 - t1) % 60))
                if True:
                    with open(log_path, 'a') as f:
                        f.write('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec \n' % (
                    epoch, t, ave_loss / (t + 1), ave_L / (t + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                t1 = time()
        model.eval()
        print("lr: ", optimizer.param_groups[0]['lr'])
        baseline.epoch_callback(model, epoch)
        scheduler.step()
        torch.save(model.state_dict(), model_save_path + '/epoch%s.pt' % (epoch))

    tt2 = time()
    print('all time, %dmin%dsec' % (
        (tt2 - tt1) // 60, (tt2 - tt1) % 60))

if __name__ == '__main__':
    dict_file = {"n_encode_layers": 4,
                 "N_samplings": 8,
                 "epochs": 400,
                 "batch": 64,
                 "batch_num": 1000,
                 "batch_verbose": 100,
                 "max_stacks": 3,
                 "max_tiers": 5,
                 "plus_tiers": 2,
                 "baseline_type": "greedy",
                 "lr": 0.0001,
                 "warmuplr": 0.001,
                 "beta": 0.1,
                 "embed_dim": 32}
    i = 0
    newpath = f'./train/Exp{i}' 
    while os.path.exists(newpath):
        i = i+1
        newpath = f'./train/Exp{i}'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    train(log_path = newpath, dict_file=dict_file)