import gym
from gym import spaces
from gym.spaces import Discrete, Tuple, MultiBinary
import numpy as np
import sys
from six import StringIO

class MyBlocksWorldEnv_2D(gym.Env):
    def __init__(self, numBlocks):
        super(MyBlocksWorldEnv_2D, self).__init__()
        self.numBlocks = numBlocks
        self.action_space = Tuple(
            [Discrete(numBlocks+1), Discrete(numBlocks+1)])
        #self.observation_space = MultiBinary([numBlocks, numBlocks])
        self.observation_space = Tuple([MultiBinary([numBlocks, numBlocks]),MultiBinary([numBlocks, numBlocks])])
        #self.start_state = []
        #self.goal_state = []
        #self.state = self.start_state
        self.numactions = (numBlocks+1)*(numBlocks+1)
        self.last_reward = 0
    def reset(self):
        self.start_state = [0,1,4,0,0,0]#start_state, 직접 설정(추후 변경해야함)
        self.goal_state = [2,3,4,5,6,0]#goal_state, 직접 설정(추후 변경해야함)
        self.state = [0,1,4,0,0,0]
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
    def ispossible_action(self, state, action):
        state = self.state2d_to_1d(state)
        block, dest = action
        if (dest<0 ) or (dest>self.numBlocks) or (block>self.numBlocks) or (state[block-1]==dest) or  ((block == dest) and (dest !=0)):
            return False
        if ((block in state) or (block==0)):
            return False
        if ((dest in state) and (dest != 0)):
            return False
        return True
    def step(self, action):
        block, dest = action
        block = int(block)
        dest = int(dest)
        self.num_run += 1
        Done = False
        reward = 0
        if(self.num_run > 200):
            Done =True
        if ((dest<0 ) or (dest>self.numBlocks) or (block>self.numBlocks) or (self.state[block-1]==dest) or  ((block == dest) and (dest !=0))):
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
                        reward = 10
                        Done = True
                    else:
                        if(inPos_before):
                            reward = -3
                        else:
                            if inPos_after:
                                reward = 0.5
                            else:
                                reward = -1
        self.last_reward = reward
        return self.get_obs(), reward, Done, None, {}
    def state1d_to_2d(self, state1d):
        l = len(state1d)
        state2d = [[0 for i in range(l)] for j in range(l)]
        for i in range(l):
            j = state1d[i]
            if j!=0:
                state2d[i][j-1] = 1
        return state2d
    def state2d_to_1d(self, state2d):
        l = len(state2d)
        state1d = [0 for i in range(l)]
        for i in range(l):
            mi = np.argmax(state2d[i])
            if state2d[i][mi] !=0:
                 state1d[i] = mi+1
            else:
                 state1d[i] = 0
        return state1d
    def get_obs(self):
        return [self.state1d_to_2d(self.state), self.state1d_to_2d(self.goal_state)]
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
