
# Monte Carlo implementation of Grid World
# With Exploring Starts

import numpy as np
import random
import pandas as pd

NO_ROWS = 3
NO_COLS = 4
WALL = [1,1]
WIN_STATE = [0, 3]
LOSE_STATE = [1, 3]
START = [2,0]
policy = "RANDOM"
actions = ["left","right","up","down"]

no_episodes = 1000


class AGENT:
    
    def __init__(self, gamma, first_visit, exploring_start):
        
        #Intialise State Values
        self.Values = np.zeros([NO_ROWS,NO_COLS])
        
        #Initialise State Action Values
        self.ActionValues = AGENT.InitialValues()
        
        #Initialise Returns
        self.Return = AGENT.InitialValues()
    
        
        for i in range(100000):
            
            #if i%1000 == 0:
            #    print(i)
                
            if exploring_start == True:
            
                START_explore = [np.random.randint(0,NO_ROWS),np.random.randint(0,NO_COLS)]
                while (START_explore == WIN_STATE) or (START_explore == LOSE_STATE) or (START_explore == WALL):
                    START_explore = [np.random.randint(0,NO_ROWS),np.random.randint(0,NO_COLS)]
                
                self.CurrentState = START_explore
            else:
                self.CurrentState = START
            
            Steps, G = AGENT.CumulativeReward(self, gamma)
        
            AGENT.ActionV_update(self,Steps, G, first_visit)
        
        self.ActionValues
        
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
    
    def ActionV_update(self, Steps, G, first_visit):
        
        if first_visit == True:
            
            Steps.reverse()
            
            G.reverse()
            
            seen = list()
            
            for i in range(len(G)):
                
                if Steps[i] not in seen:
                    
                    seen.append(Steps[i])
                    #n = self.ActionValues[str(Steps[i][0])]['Count']
                    n = 10
        
                    self.ActionValues[str(Steps[i][0])]['actions'][Steps[i][1]] = self.ActionValues[str(Steps[i][0])]['actions'][Steps[i][1]] + (1/(n+1))*(G[i] - self.ActionValues[str(Steps[i][0])]['actions'][Steps[i][1]])
                    self.ActionValues[str(Steps[i][0])]['Count'] = n+1
                else:
                    pass
            
        else:
            
            for i in range(len(G)):
                
                n = self.ActionValues[str(Steps[i][0])]['Count']
        
                self.ActionValues[str(Steps[i][0])]['actions'][Steps[i][1]] = self.ActionValues[str(Steps[i][0])]['actions'][Steps[i][1]] + (1/(n+1))*(G[i] - self.ActionValues[str(Steps[i][0])]['actions'][Steps[i][1]])
                self.ActionValues[str(Steps[i][0])]['Count'] = n+1
            
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
        
    def CumulativeReward(self, gamma):
        
        G = 0
        
        Steps = list()
        R = list()
        
        while ((self.CurrentState != WIN_STATE) and (self.CurrentState != LOSE_STATE)):
            
            act = AGENT.Policy(self,self.CurrentState,0.1)
            
            #Same state and action
            Steps.append([self.CurrentState, act])
            
            self.CurrentState = AGENT.Next_State(self,self.CurrentState,act)
            
            R.append(AGENT.Reward(self,self.CurrentState))
          
        # Calculate reward for each of the visited states
        
        G = list()
        R.reverse()
        Steps.reverse()

        G_val = R[0]
        
        # Calculate reward for each visited state
        for i in range(len(R)-1):
            
            G_val = gamma*G_val + R[i+1]
            G.append(G_val)
                
        return Steps, G
    
        
first_visit = False
exploring_start = True
gamma = 0.9
ag = AGENT(gamma, first_visit, exploring_start)
print(ag.ActionValues)
