import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from datetime import datetime

from model import AttentionModel
from baseline import RolloutBaseline
from data import generate_data, Generator

def train(log_path = None):
    #将会让程序在开始torch.save(model.state_dict(), '%s%s_epoch%s.pt' % (cfg.weight_dir, cfg.task, epoch))时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    # 进而实现网络的加速
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    log_path = './sample.txt'
    start_t=time()
    # open w就是覆盖，a就是在后面加 append
    with open(log_path, 'w') as f:
        f.write(datetime.now().strftime('%y%m%d_%H_%M'))
    with open(log_path, 'a') as f:
        f.write('\n start training \n')

    model = AttentionModel(device=device, max_stacks = 3, max_tiers = 5, n_containers = 9)
    model.train()
    model=model.to(device)

    baseline = RolloutBaseline(model, task=None,  weight_dir = None, log_path=log_path, max_stacks = 3, max_tiers = 5, n_containers = 9)

    optimizer = optim.Adam(model.parameters(), lr=.001)

    #bs batch steps number of samples = batch * batch_steps
    def rein_loss(model, inputs, bs, t, device):
        # ~ inputs = list(map(lambda x: x.to(device), inputs))

        # decode_type是贪心找最大概率还是随机采样
        # L(batch) 就是返回的cost ll就是采样得到的路径的概率
        L, ll = model(inputs, decode_type='sampling')
        #b = bs[t] if bs is not None else baseline.eval(inputs, L)
        b=torch.FloatTensor([L.mean()]).to(device)
        #return ((L - b) * ll).mean(), L.mean()
        return ((L-b)*ll).mean(), L.mean()

    tt1 = time()


    t1=time()
    epochs = 10
    batch = 64
    batch_verbose = 64
    for epoch in range(epochs):

        ave_loss, ave_L = 0., 0.

        datat1=time()
        dataset=Generator(device, n_samples=64*1000, n_containers=9, max_stacks=3, max_tiers=5)
        datat2=time()
        print('data_gen: %dmin%dsec' % ((datat2 - datat1) // 60, (datat2 - datat1) % 60))

        bs=baseline.eval_all(dataset)
        bs = bs.view(-1, batch) if bs is not None else None  # bs: (cfg.batch_steps, cfg.batch) or None

        model.train()
        dataloader=DataLoader(dataset, batch_size = batch, shuffle = True)
        for t, inputs in enumerate(dataloader):
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
                print('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec' % (
                    epoch, t, ave_loss / (t + 1), ave_L / (t + 1), (t2 - t1) // 60, (t2 - t1) % 60))
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