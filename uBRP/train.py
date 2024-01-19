import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from datetime import datetime
import os
from model import AttentionModel
from baseline import RolloutBaseline
from data import generate_data, Generator

def train(log_path = None):
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.device(0))
    if device!='cpu':
        torch.cuda.set_device(device)
    print(train)
    n_encode_layers = 4
    N_samplings = 8
    log_path = f'./NoAug_x{N_samplings}_Linearx2Init_{n_encode_layers}_layers_0.txt'

# 파일이 이미 존재하는지 확인
    if os.path.exists(log_path):
        # 파일이 이미 존재한다면 새로운 이름 생성
        base, ext = os.path.splitext(log_path)
        counter = 0
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        log_path = f"{base}_{counter}{ext}"
    start_t=time()
    # open w就是覆盖，a就是在后面加 append
    with open(log_path, 'w') as f:
        f.write(datetime.now().strftime('%y%m%d_%H_%M'))
    with open(log_path, 'a') as f:
        f.write('\n start training \n')
    
    model = AttentionModel(device=device, n_encode_layers=n_encode_layers, max_stacks = 3, max_tiers = 5, n_containers = 9)
    model=model.to(device)
    model.train()

    baseline = RolloutBaseline(model, task=None,  device=device,weight_dir = None, log_path=log_path, max_stacks = 3, max_tiers = 5, n_containers = 9)

    optimizer = optim.Adam(model.parameters(), lr=.0001)

    #bs batch steps number of samples = batch * batch_steps
    def rein_loss(model, inputs, bs, t, device):
        # ~ inputs = list(map(lambda x: x.to(device), inputs))

        # decode_type是贪心找最大概率还是随机采样
        # L(batch) 就是返回的cost ll就是采样得到的路径的概率
        with torch.no_grad():
            model.eval()
            bLs = torch.zeros([batch]).to(device)
            for i in range(N_samplings):
                bL, bll = model(inputs, decode_type='sampling')
                bLs = bLs + bL
            b = bLs/N_samplings
        model.train()
        L, ll = model(inputs, decode_type='sampling')
        #b = bs[t] if bs is not None else baseline.eval(inputs, L)
        return ((L - b) * ll).mean(), L.mean()
        #return ((L-bL)*ll).mean(), L.mean()

    tt1 = time()


    t1=time()
    epochs = 30
    batch = 128
    batch_verbose = 1
    for epoch in range(epochs):

        ave_loss, ave_L = 0., 0.

        datat1=time()
        max_stacks = 3
        max_tiers = 5
        dataset=Generator(device, n_samples=128*50, n_containers=9, max_stacks=max_stacks, max_tiers=max_tiers)
        datat2=time()
        print('data_gen: %dmin%dsec' % ((datat2 - datat1) // 60, (datat2 - datat1) % 60))

        bs=baseline.eval_all(dataset)
        bs = bs.view(-1, batch) if bs is not None else None  # bs: (cfg.batch_steps, cfg.batch) or None

        model.train()
        dataloader=DataLoader(dataset, batch_size = batch, shuffle = True)
        for t, inputs in enumerate(dataloader):
            #print(inputs.size())
            loss,L_mean=rein_loss(model,inputs,bs,t,device)
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()

            # print('grad: ', model.Decoder.Wk1.weight.grad[0][0])
            # https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()

            ave_loss += loss.item()
            ave_L += L_mean.item()

            if t % (batch_verbose) == 0:
                t2 = time()
                # //60是对小数取整
                print('Epoch %d (batch = %d): Loss: %1.3f avg_L: %1.3f, batch_Loss: %1.3f batch_L: %1.3f %dmin%dsec' % (
                    epoch, t, ave_loss / (t + 1), ave_L / (t + 1), loss.item(), L_mean.item(), (t2 - t1) // 60, (t2 - t1) % 60))
                # 如果要把日志文件保存下来
                if True:
                    with open(log_path, 'a') as f:
                        f.write('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec \n' % (
                    epoch, t, ave_loss / (t + 1), ave_L / (t + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                t1 = time()

        #print('after this epoch grad: ', model.Decoder.Wk1.weight.grad[0][0])

        # 看是不是要更新baseline
        #这里为了让baseline不变化给model加上eval
        model.eval()
        baseline.epoch_callback(model, epoch)
        torch.save(model.state_dict(), './epoch%s.pt' % (epoch))

        if epoch==epochs-1:
            #data = data_from_txt("data/test.txt")
            #data=data.to(device)
            #baseline.model.eval()
            #torch.save(baseline.model.state_dict(), '%s%s_epoch%s_2.pt' % (cfg.weight_dir, cfg.task, epoch))
            #torch.save(baseline.model.Decoder.Encoder.state_dict(),'%s%s_encoder_epoch%s.pt' % (cfg.weight_dir, cfg.task, epoch))
            #with torch.no_grad():
                #cost=baseline.rollout(model=baseline.model,dataset=data,batch=40)
            #print('test baseline model')
            #print('test.txt:mean',cost.mean())
            pass

    tt2 = time()
    print('all time, %dmin%dsec' % (
        (tt2 - tt1) // 60, (tt2 - tt1) % 60))

if __name__ == '__main__':
    train()
    #nohup python -u train.py -p Pkl/CRP_9_3_5_train.pkl >>./Csv/nohup.txt 2>&1 &