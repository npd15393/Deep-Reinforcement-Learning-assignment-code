#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import time
for i in range(7):
    WORLD_SIZE = 5*(i+1)
    A_POS = [0, 0]
    A_PRIME_POS = [4, 1]
    B_POS = [WORLD_SIZE-1, WORLD_SIZE-1]
    B_PRIME_POS = [2, 3]
    discount = 0.9
    
    world = np.zeros((WORLD_SIZE, WORLD_SIZE))
    
    # left, up, right, down
    actions = ['L', 'U', 'R', 'D']
    
    actionProb = []
    for i in range(0, WORLD_SIZE):
        actionProb.append([])
        for j in range(0, WORLD_SIZE):
            actionProb[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))
    
    nextState = []
    actionReward = []
    for i in range(0, WORLD_SIZE):
        nextState.append([])
        actionReward.append([])
        for j in range(0, WORLD_SIZE):
            next = dict()
            reward = dict()
         
            if i == 0:
                next['U'] = [i, j]
                reward['U'] = -1.0
            else:
                next['U'] = [i - 1, j]
                reward['U'] = -1.0
    
            if i == WORLD_SIZE - 1:
                next['D'] = [i, j]
                reward['D'] = -1.0
            else:
                next['D'] = [i + 1, j]
                reward['D'] = -1.0
    
            if j == 0:
                next['L'] = [i, j]
                reward['L'] = -1.0
            else:
                next['L'] = [i, j - 1]
                reward['L'] = -1.0
    
            if j == WORLD_SIZE - 1:
                next['R'] = [i, j]
                reward['R'] = -1.0
            else:
                next['R'] = [i, j + 1]
                reward['R'] = -1.0
    
#            if [i, j] == A_POS:
#                next['L'] = next['R'] = next['D'] = next['U'] = A_PRIME_POS
#                reward['L'] = reward['R'] = reward['D'] = reward['U'] = 10.0
#    
#            if [i, j] == B_POS:
#                next['L'] = next['R'] = next['D'] = next['U'] = B_PRIME_POS
#                reward['L'] = reward['R'] = reward['D'] = reward['U'] = 5.0
    
            nextState[i].append(next)
            actionReward[i].append(reward)
    
    start_time=time.time()
    # for figure 3.5
    while True:
        # keep iteration until convergence
        newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
        for i in range(0, WORLD_SIZE):
            for j in range(0, WORLD_SIZE):
                for action in actions:
                    newPosition = nextState[i][j][action]
                    # bellman equation
                    if (newPosition[0] == 0 and newPosition[1] == 0) or (newPosition[0] == WORLD_SIZE - 1 and newPosition[1] == WORLD_SIZE - 1):
                        val=0
                    else:
                        val=world[newPosition[0], newPosition[1]]
                    newWorld[i, j] += actionProb[i][j][action] * (actionReward[i][j][action] + discount * val)
        if np.sum(np.abs(world - newWorld)) < 1e-4:
            print('Random Policy')
            print(newWorld)
            break
        world = newWorld
    print("Time for random policy = {0} for size={1}".format(time.time()-start_time,WORLD_SIZE))

#    start_time1=time.time()
#    # for figure 3.8
#    world = np.zeros((WORLD_SIZE, WORLD_SIZE))
#    while True:
#        # keep iteration until convergence
#        newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
#        for i in range(0, WORLD_SIZE):
#            for j in range(0, WORLD_SIZE):
#                values = []
#                for action in actions:
#                    newPosition = nextState[i][j][action]
#                    # value iteration
#                    values.append(actionReward[i][j][action] + discount * world[newPosition[0], newPosition[1]])
#                newWorld[i][j] = np.max(values)
#        if np.sum(np.abs(world - newWorld)) < 1e-4:
#            print('Optimal Policy')
#            print(newWorld)
#            break
#        world = newWorld
#    print("Time for optimal policy = {0}".format(time.time()-start_time1))

