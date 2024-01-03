import gym
from gym import spaces
from gym.spaces import Discrete, Tuple
import numpy as np
import sys
from six import StringIO
def state1d_to_2d(state1d):
    l = len(state1d)
    state2d = [[0 for i in range(l)] for j in range(l)]
    for i in range(l):
        j = state1d[i]
        if j!=0:
            state2d[i][j-1] = 1
    return state2d
class MyBlocksWorldEnv(gym.Env):
    def __init__(self, numBlocks):
        super(MyBlocksWorldEnv, self).__init__()
        self.numBlocks = numBlocks
        self.action_space = Tuple(
            [Discrete(numBlocks+1), Discrete(numBlocks+1)])
        self.observation_space = Tuple(
            [Discrete(numBlocks) for i in range(numBlocks*2)])
        #self.start_state = []
        #self.goal_state = []
        #self.state = self.start_state
        self.numactions = (numBlocks+1)*(numBlocks+1)
        self.last_reward = 0
    def reset(self):
        self.start_state = [0,3,0,5,0,1,0]
        self.goal_state = [0,1,2,3,4,5,6]
        self.state = [0,3,0,5,0,1,0]
        self.num_run = 0
        return self.get_obs()
    def in_pos(self, block_id):
        s = self.state
        g = self.goal_state
        block = block_id-1
        if(s[block] == g[block]):
            if(s[block] == 0):
                return True
            else:
                return self.in_pos(s[block])
        else:
            return False
    def step(self, action):
        block, dest = action
        self.num_run += 1
        Done = False
        reward = 0
        if(self.num_run > 50):
            Done =True
        if ((dest<0) or (dest>self.numBlocks) or (block>self.numBlocks) or (self.state[block-1]==dest) or  ((block == dest) and (dest !=0))):
            # 잘못된 움직임일경우
            reward = -10 #SHOULD BE MODIFIED
        else:
            if ((block in self.state) or (block==0)):
                #Block 위에 무언가 있을 경우 -> 가로막혀 불가능한 움직임
                reward = -10 #SHOULD BE MODIFIED
            else:
                if ((dest in self.state) and (dest != 0)):
                    reward = -10 #SHOULD BE MODIFIED
                else:
                    #MOVE
                    inPos_before = self.in_pos(block) # Should be implemented
                    self.state [block-1] = dest
                    inPos_after = self.in_pos(block) # Should be implemented
                    if(str(self.state) == str(self.goal_state)):
                        reward = 1
                        Done = True
                    else:
                        if(inPos_before):
                            reward = -2
                        else:
                            if inPos_after:
                                reward = 0.5
                            else:
                                reward = -1
        self.last_reward = reward
        return self.get_obs(), reward, Done, None, {}
    def get_obs(self):
        return np.concatenate((self.state, self.goal_state))
    def render(self):
        outfile = sys.stdout
        outfile.write("************** New Step ***************\n") 
        outfile.write("[block_to_move, destination]; b [0,numBlocks-1], d[0,numBlocks]\n")                         
        outfile.write("Initial state: " + str(self.start_state)+ "\n")
        outfile.write("Current state: " + str(self.state)+ "\n")
#       outfile.write (str(self.state))
        outfile.write("Goal state:    "+ str(self.goal_state) + "\n")
        outfile.write ("Reward: " + str(self.last_reward)+ "\n")
        return outfile
