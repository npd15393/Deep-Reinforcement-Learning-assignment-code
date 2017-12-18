import gym
import itertools
from keras.models import Sequential
from keras.layers import Dense,Dropout,Input,merge,Reshape
from keras.backend import repeat_elements, sum
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import random as rand
from AC_lib import AC
#
#rewards=[]
#
#def create_model(env):
#        learning_rate=0.0001
#        #Create a nn of 4 fc and dropout layers
#        model = Sequential()
#        model.add(Dense(32, activation="relu", input_dim=4))
#        #model.add(Dropout(0.1))
#        model.add(Dense(32, activation="relu"))
##       # model.add(Dropout(0.1))
#        model.add(Dense(32, activation="relu"))
##        #model.add(Dropout(0.1))
#        model.add(Dense(2, activation='linear'))
#        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))
#        print("NN Architecture created")
#        return model
#
#
#def running_policy(epsi,q):
#        if np.random.rand()<=max([epsi,0.01]):
#            return rand.choice([0,1])
#        return np.argmax(q)
#
#
#def QLearn(envr,model,nE,pR=False):
#        #Q= defaultdict(lambda: np.zeros(envr.action_space.n))
#    
#        assert not (model == None or envr == None)
#        
#        rb=new_qlearn_epoch(envr,model,nE,True)
#        model=new_qlearn_epoch(envr,model,nE,False,rb)
#        
#        return model
# 
#    
#def new_qlearn_epoch(envr,model,nE,refill=False,rb=[],buffersize=64):
#        epsilon=1
#        GAMMA=0.99
#        temp_buffer=[]
#
#        for i in range(nE):  
#            
#            current_pos=envr.reset()
#            done=False
#            treward=0
#            
#            while not done:        
#                
#                #envr._render()
#                q=model.predict(np.reshape(current_pos,(1,4)))[0]
#                
#                epsilon*=0.99
#                if not refill:
#                    act=running_policy(epsilon,q)
#                else:
#                    act=rand.choice([0,1])
#                
#                next_state, reward, done, _= envr.step(act)
#                #reward=reward if not done else -100
#                
#                treward+=reward                
#                
#                q1=model.predict(np.reshape(next_state,(1,4)))[0]
#                target= model.predict(np.reshape(current_pos,(1,4)))[0]
#                    
#                if done:
#                                target[act]=reward
#                                rewards.append(treward)
#                            
#                else:
#                                target[act]=reward+GAMMA*np.amax(q1)
#                            
#                #model.fit(np.reshape(current_pos,(1,4)),np.reshape(target,(1,2)),epochs=1,verbose=0)
#                temp_buffer.append([current_pos,act,next_state,target])
#                
#                if done and (i+1)%10==0:
#                    print('Episode:{0}/{2} - Total reward={1}'.format(i+1,treward,nE))
#                                 
#                current_pos=next_state
#                
#            if len(temp_buffer)>=buffersize:
#                    if not refill:
#                        #rbs=rand.sample(range(len(rb)), 32 )
#                        
#                        X=[]
#                        targets=[]
#                        
#                        while len(rb)>=1:
#                            u=fetch_random_value(rb)
#                            X.append(u[0])
#                            targets.append(u[3])
#                            
#                            model.fit(np.reshape(u[0],(1,4)),np.reshape(u[3],(1,2)),epochs=1,verbose=0)
#                        rb=copy_buffer(temp_buffer)
#                        temp_buffer=[]
#                        
#                    else:
#                        return temp_buffer
#                        print('Initial trial done')
#
#                    
#                
#        
#        return model
#    
#def fetch_random_value(rb):
#    u=rand.choice(range(len(rb)))
#    t=rb[u]
#    rb.pop(u)
#    return t
#
#def copy_buffer(rb):
#            nb=[]
#            for i in rb:
#                nb.append(i)
#            return nb

#def run_policy(self,env,model):
#            current_pos=env.reset()
#            q=model.predict(current_pos)
#            
#            act=np.argmax(q)
#            for t in itertools.count():        
#                next_state, reward, done, _= env.step(act)
#                if done:
#                    break
#                else:
#                    current_pos=next_state
#                                
env=gym.make('CartPole-v1')
#model=QLearn(env,create_model(env),1000)
ai=AC(env,2,4)
model=ai.train(1000,500)
model.save('cp.h5')
plt.plot(range(len(rewards)),rewards)