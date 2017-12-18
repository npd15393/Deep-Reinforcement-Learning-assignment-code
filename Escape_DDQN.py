# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:39:00 2017

@author: Nishant
"""

import gym
import itertools
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import random as rand
from gridworld import gameEnv

rewards=[]

def create_model(env):
        learning_rate=0.001
        input_shape = (84, 84, 3)
        #Create a nn of 4 fc and dropout layers
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                 activation='relu',
                 input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (4, 4),strides=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(nA, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
        print("NN Architecture created")
        return model


def running_policy(epsi,q):
        if np.random.rand()<=max([epsi,0.01]):
            return rand.choice(range(4))
        return np.argmax(q)


def DDQLearn(envr,model,target,nE,pR=False):
        #Q= defaultdict(lambda: np.zeros(envr.action_space.n))
    
        assert not (model == None or envr == None)
        
        rb=new_qlearn_epoch(envr,model,target,nE,True)
        print('now training')
        model=new_qlearn_epoch(envr,model,target,nE,False,rb)
        
        return model
 
    
def new_qlearn_epoch(envr,model,targetnn,nE,refill=False,rb=[],buffersize=128):
        epsilon=1
        GAMMA=0.99
        temp_buffer=[]
        tau = 0.1
        for i in range(nE):  
            
            current_pos=envr.reset()
            done=False
            treward=0
            
            for j in range(50):        
                
                #envr.renderEnv()
                q=model.predict(np.reshape(current_pos,(1,84,84,3)))[0]
                
                epsilon*=0.99
                if not refill:
                    act=running_policy(epsilon,q)
                else:
                    act=rand.choice(range(4))
                
                next_state, reward, done= envr.step(act)
                #reward if not done else -100
                treward+=reward                
                #model.fit(np.reshape(current_pos,(1,4)),np.reshape(target,(1,2)),epochs=1,verbose=0)
                temp_buffer.append([current_pos,act,next_state])
                print(str(act)+" "+str(reward))
                current_pos=next_state
                
            
            rewards.append(treward)
            if i%500==0:
                print('Episode:{0}/{2} - Total reward={1}'.format(i+1,treward[],nE))
                model.save('EscapeAIModel.h5')
            
            if len(temp_buffer)>=buffersize:
                    if not refill:
                        #rbs=rand.sample(range(len(rb)), 32 )
                        
                        X=[]
                        targets=[]
                        sb=rand.sample(rb,32)
                        print()
                        for u in sb:
                            X.append(u[0])
                            
                            q1=targetnn.predict(np.reshape(u[2],(1,84,84,3)))[0]
                            target= model.predict(np.reshape(u[0],(1,84,84,3)))[0]
                    
                            if done:
                                target[u[1]]=reward                          
                            else:
                                target[act]=reward+GAMMA*np.amax(q1)
                            
                            targets.append(target)                          
                        
                        model.fit(np.reshape(X,(32,84,84,3)),np.reshape(targets,(32,nA)),epochs=1,verbose=0)
                        targetupdates=model.predict(np.reshape(X,(32,84,84,3)))
                        if rand.random<tau:
                            targetnn.fit(np.reshape(X,(32,84,84,3)),np.reshape(targetupdates,(32,nA)),epochs=1,verbose=0)
                        rb=copy_buffer(temp_buffer)
                        temp_buffer=[]
                        
                    else:
                        return temp_buffer
                        print('Initial trial done')
        return model

def copy_buffer(rb):
            nb=[]
            for i in rb:
                nb.append(i)
            return nb

def run_policy(env,model):
            current_pos=env.reset()
            q=model.predict(current_pos)
            
            act=np.argmax(q)
            for t in itertools.count():        
                next_state, reward, done, _= env.step(act)
                if done:
                    break
                else:
                    current_pos=next_state
                                
env=gameEnv(False,5)
nS=(1,84,84,3)
nA=4
model=DDQLearn(env,create_model(env),create_model(env),100000)
plt.plot(range(len(rewards)),rewards)
model.save('EscapeAIModel.h5')