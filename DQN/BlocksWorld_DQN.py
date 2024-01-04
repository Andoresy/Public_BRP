import gym
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
num_blocks = 6
gym.register(
    id='BlocksWorld-v1',
    entry_point='gym.envs.classic_control:MyBlocksWorldEnv_2D', 
    kwargs={"numBlocks":num_blocks} 
)
env = gym.make('BlocksWorld-v1', num_blocks)
obs = env.reset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def integer_to_action (int_action):
# From an integer returns the encoded format for an action
# [block to move, destination]
    ret = []
    ret.append(int(int_action/(num_blocks+1)))
    ret.append(int_action%(num_blocks+1))
    return ret
def action_to_integer (action):
# From an integer returns the encoded format for an action
# [block to move, destination]
    return action[0]*(num_blocks+1)+action[1]
"""Transition"""
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

"""Replay Memory""" #By Pytorch Tutorial
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size = 3, stride = 1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size = 3, stride =  1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.layer1 = nn.Linear(32*h*w, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3
n_actions = env.numactions
# 상태 관측 횟수를 얻습니다.
state = env.reset()
n_observations = len(state)
policy_net = DQN(num_blocks, num_blocks, n_actions).to(device)
target_net = DQN(num_blocks, num_blocks, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(500)


steps_done=0
def select_action_acceptable(state):
    dstate = state[0][0]
    possible_actions = []
    for i in range(env.numactions):
        action = integer_to_action(i)
        if env.ispossible_action(dstate, action):
            possible_actions.append(i)
    possible_values = [[policy_net(state)[0][i] if i in possible_actions else -999999 for i in range(env.numactions)]]
    return torch.tensor(possible_values).max(1)[1].item()
def select_action_acceptable_byrandom(state):
    state = state[0][0]
    sample_action = env.action_space.sample()
    while not env.ispossible_action(state, sample_action):
        sample_action = env.action_space.sample()
    return sample_action
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return torch.tensor([[select_action_acceptable(state)]],device=device, dtype=torch.long)
    else:
        return torch.tensor([[action_to_integer(select_action_acceptable_byrandom(state))]], device=device, dtype=torch.long)
episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.\
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 100

for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    state= env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0
    for t in count():
        action = select_action(state)
        #print(action)
        observation, reward, terminated, truncated, _ = env.step(integer_to_action(action[0][0]))
        episode_reward+=reward
        reward = torch.tensor([reward], device=device)
        if i_episode==99:
            print(env.state, action, reward)
            print(env.goal_state)
        done = terminated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        optimize_model()

        # 목표 네트워크의 가중치를 소프트 업데이트
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            print("episode reward: ", episode_reward)
            break
print(episode_durations)
print('Complete')