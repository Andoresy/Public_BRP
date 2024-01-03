import gym
import numpy as np
num_blocks = 7
gym.register(
    id='BlocksWorld-v1',
    entry_point='gym.envs.classic_control:MyBlocksWorldEnv', 
    kwargs={"numBlocks":num_blocks} 
)
env = gym.make('BlocksWorld-v1', num_blocks)
obs = env.reset()


Q=np.zeros([(num_blocks+1)**(num_blocks), (num_blocks+1)**2])#Very Large
learning_rate=0.1 #Learning Rate
lamda=0.9 #Discount Rate
def integer_to_action (int_action):
# From an integer returns the encoded format for an action
# [block to move, destination]
    ret = []
    ret.append(int(int_action/(num_blocks+1)))
    ret.append(int_action%(num_blocks+1))
    return ret
def state_id(state):
    state = state[:num_blocks]
    s_id = 0
    for i in range(num_blocks):
        s_id*=(num_blocks+1)
        s_id+=state[i]
    return s_id
def action_fromid(a_id):
    state = []
    for i in range(2):
        state.insert(0, a_id%(num_blocks+1))
        a_id = a_id//(num_blocks+1)
    return state
n_episode= 400
length_episode=400

#최적 행동 가치 함수 찾기
training_steps = []
for i in range(n_episode):
    s=env.reset()
    s=state_id(s)
    steps = 0
    for j in range(length_episode):
        argmaxs= np.argwhere(Q[s,:]==np.amax(Q[s,:])).flatten().tolist()
        a=np.random.choice(argmaxs)
        s1, r, done,_,_=env.step(integer_to_action(a))
        s1 = state_id(s1)
        Q[s,a] = Q[s,a]+learning_rate*(r+lamda*np.max(Q[s1,:])-Q[s,a])
        s=s1
        steps+=1
        if done:
            break
    training_steps.append(steps)
np.set_printoptions(precision=2)
#print(Q)
print(training_steps)