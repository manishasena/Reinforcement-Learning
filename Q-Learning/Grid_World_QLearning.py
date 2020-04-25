
# Q Learning implementation of Grid World
# With Exploring Starts

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

NO_ROWS = 3
NO_COLS = 4
WALL = [1,1]
WIN_STATE = [0, 3]
LOSE_STATE = [1, 3]
START = [2,0]
policy = "RANDOM"
actions = ["left","right","up","down"]


class AGENT:
    
    def __init__(self, gamma, first_visit, exploring_start, no_episodes,epsilon, alpha):
        
        #Intialise State Values
        self.Values = np.zeros([NO_ROWS,NO_COLS])
        
        #Initialise State Action Values
        self.ActionValues = AGENT.InitialValues()
        
        #Initialise Returns
        self.Return = AGENT.InitialValues()
    
        # Number of steps in epsiode:
        self.n = list()
        
        for i in range(no_episodes):
            
            if i%1000 == 0:
                print(i)
                
            if exploring_start == True:
            
                START_explore = [np.random.randint(0,NO_ROWS),np.random.randint(0,NO_COLS)]
                while (START_explore == WIN_STATE) or (START_explore == LOSE_STATE) or (START_explore == WALL):
                    START_explore = [np.random.randint(0,NO_ROWS),np.random.randint(0,NO_COLS)]
                
                self.CurrentState = START_explore
            else:
                self.CurrentState = START
            
            self = AGENT.CumulativeReward(self, gamma,epsilon)
            
        
        for i in self.ActionValues:
            if ((i != str(WIN_STATE)) and (i != str(LOSE_STATE)) and (i != str(WALL))):
                print([i,AGENT.Policy(self,i,0)])
        #print(s.ActionValues)
                
    def InitialValues():
        states = list() #np.zeros([NO_ROWS*NO_COLS,2])
        n = -1
        for i in range(NO_ROWS):
            for j in range(NO_COLS):
                n += 1
                states.append([i,j])
                #states[n,0] = i
                #states[n,1] = j
        
        AV = {}
        for i in states:
            AV[str(i)] = {}
            AV[str(i)]['actions'] = {}
            
            for j in actions:
                
                AV[str(i)]['actions'][j] = 0
                
            AV[str(i)]['Count'] = 0
                
        return AV
                
    def Policy(self,S,epsilon):
        
        possible_actions = self.ActionValues[str(S)]['actions']
        
        if np.random.uniform() <= epsilon:
            
            action_only = list(possible_actions.keys())
            act = random.choice(action_only)
            
        else:
            
            values = list(possible_actions.values())
            
            max_val = max(values)
            
            res_list = [i for i, value in enumerate(values) if value == max_val]
            action_only = list(possible_actions.keys())
            move_choices = [action_only[i] for i in res_list] 
        
            act = random.choice(move_choices)
        
        return act
    
    def ActionV_update(self, Steps, R, alpha):
            
        max_Q = -10000
        for j in self.ActionValues[str(self.CurrentState)]['actions']:
            
            if self.ActionValues[str(self.CurrentState)]['actions'][j] > max_Q:
                max_Q = self.ActionValues[str(self.CurrentState)]['actions'][j]
                
        self.ActionValues[str(Steps[0])]['actions'][Steps[1]] = self.ActionValues[str(Steps[0])]['actions'][Steps[1]] + alpha*(R + gamma*max_Q - self.ActionValues[str(Steps[0])]['actions'][Steps[1]])
            
        return self
    
    def Next_State(self,S,act):

        S_nxt = S.copy()
        
        if act == 'left':
            S_nxt[1] = S_nxt[1]-1
        elif act == 'right':
            S_nxt[1] = S_nxt[1]+1
        elif act == 'up':
            S_nxt[0] = S_nxt[0]-1
        else:
            S_nxt[0] = S_nxt[0]+1

        if (S_nxt == WALL) or (S_nxt[0] <0 or S_nxt[0] >= 3) or (S_nxt[1] <0 or S_nxt[1] >= 4):
            S_nxt = S
        
        return S_nxt
    
    def Reward(self,S):
        
        if S == WIN_STATE:
            R = 1
        elif S == LOSE_STATE:
            R = -1
        else:
            R = 0
            
        return R
        
    def CumulativeReward(self, gamma, epsilon):
        
        Steps = list()
        G = list()

        n = 0
        
        while ((self.CurrentState != WIN_STATE) and (self.CurrentState != LOSE_STATE)):
            
            n += 1
            act = AGENT.Policy(self,self.CurrentState,epsilon)
            
            #Same state and action
            Steps = [self.CurrentState, act]
            
            # NExt state
            self.CurrentState = AGENT.Next_State(self,self.CurrentState,act)
            
            #Reward
            R = AGENT.Reward(self,self.CurrentState)
            
            # Update Action Values
            self = AGENT.ActionV_update(self, Steps, R, alpha)
        
        self.n.append(n)

                
        return self
    
no_episodes = 100
first_visit = False
exploring_start = False
gamma = 0.9
epsilon = 0.1
alpha = 0.1
no_trials = 5

# Store number of steps from each trial
n_holder = np.zeros([no_trials,no_episodes])

for j in range(no_trials):
    
    ag = AGENT(gamma, first_visit, exploring_start, no_episodes,epsilon, alpha)
    n_holder[j,:] = ag.n

#Plotting
plt.plot(np.mean(n_holder,axis=0))
plt.title('Q-Learning Performance')
plt.xlabel('Epsiodes')
plt.ylabel('Steps to reach WIN STATE')
plt.show()

