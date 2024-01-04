def integer_to_action (int_action):
# From an integer returns the encoded format for an action
# [block to move, destination]
    ret = []
    ret.append(int(int_action/(4+1)))
    ret.append(int_action%(4+1))
    return ret
def action_to_integer (action):
# From an integer returns the encoded format for an action
# [block to move, destination]
    return action[0]*(4+1)+action[1]
a = [2,3]
def state1d_to_2d( state1d):
        l = len(state1d)
        state2d = [[0 for i in range(l)] for j in range(l)]
        for i in range(l):
            j = state1d[i]
            if j!=0:
                state2d[i][j-1] = 1
        return state2d
import numpy as np
def state2d_to_1d( state2d):
        l = len(state2d)
        state1d = [0 for i in range(l)]
        for i in range(l):
            mi = np.argmax(state2d[i])
            if state2d[i][mi] !=0:
                 state1d[i] = mi+1
            else:
                 state1d[i] = 0
        return state1d
print(state1d_to_2d([2,3,0]))
print(state2d_to_1d(state1d_to_2d([2,3,0])))
