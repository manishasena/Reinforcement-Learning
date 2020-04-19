import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

WIN_STATE = 1001
LOSE_STATE = 0
actions = ["left","right"]
no_states = 1002
START = 500
jump_size = 100
no_features = 2

class AGENT:
    
    def __init__(self, epsilon, iterations, alpha, gamma):

        self.weights = np.zeros([int((WIN_STATE-1-LOSE_STATE)//no_aggregation),1])
        
        self.mu = np.zeros([int((WIN_STATE-1-LOSE_STATE)//no_aggregation),1])

        for i in range(iterations):
            
            self.CurrentState = START
            
            steps, G = AGENT.Cumulative_Reward(self, gamma)
            self = AGENT.mu_Update(self, steps)
            self = AGENT.Weight_Update(self, steps, G, alpha)
            # Update state values
        
        self = AGENT.StateValue_update(self)
        
        
    
    def StateValue_update(self):

        self.StateValues = np.zeros([int(WIN_STATE-1-LOSE_STATE),2])

        for i in range(int(WIN_STATE-1-LOSE_STATE)):
            self.StateValues[i,0] = i + 1
            self.StateValues[i,1] = self.weights[int(i//no_aggregation)]

        return self
    
    def Policy(self, epsilon):
        
        act = random.choice(actions)
        hops = random.randint(1,(jump_size+1))
        
        return act, hops
    
    def Next_State(self, act, hops):
        
        if act == "left":
            
            Next_state = self.CurrentState - hops
            
            if Next_state <= 0:
                Next_state = LOSE_STATE

            
        elif act == "right":
            
            Next_state = self.CurrentState + hops
            
            if Next_state >= (no_states-1):
                Next_state = WIN_STATE
            
        self.CurrentState = Next_state
        
        return self
    
    def Reward(self):
        
        if self.CurrentState == WIN_STATE:
            R = 1
        elif self.CurrentState == LOSE_STATE:
            R = -1
        else:
            R = 0
            
        return R
    
    def mu_Update(self, steps):
        
        for i in range(len(steps)):
            
            self.mu[int((steps[i]-1)//no_aggregation)] += 1
        
        return self
        
    
    
    def Cumulative_Reward(self, gamma):
        
        steps = list()
        R = list()
        
        while (self.CurrentState != WIN_STATE) and (self.CurrentState != LOSE_STATE):
            
                # Add state to list
                steps.append(self.CurrentState)
                
                act, hops = AGENT.Policy(self,epsilon)
                self = AGENT.Next_State(self,act,hops)
                
                R.append(AGENT.Reward(self))
                
        G = list()
        R.reverse()
        steps.reverse()


        G_val = R[0]
        G.append(G_val)
        
        # Calculate reward for each visited state
        for i in range(len(R)-1):
            
            G_val = gamma*G_val + R[i+1]
            G.append(G_val)
        
        return steps, G


    def Weight_Update(self,steps,G,alpha):
        

        for i in range(len(steps)):
            
            x = np.zeros([int((WIN_STATE-1-LOSE_STATE)//no_aggregation),1]) 
            x[int((steps[i]-1)//no_aggregation)] = 1
            v_hat = self.weights*x
            
            mu = self.mu[int((steps[i]-1)//no_aggregation)]/np.sum(self.mu)
            
            self.weights = self.weights + mu*alpha*(G[i] - v_hat)*x

        
        return self
    

no_aggregation = 100
iterations = 10000
epsilon = 0.1
alpha = 0.05
gamma = 1
ag = AGENT(epsilon,iterations, alpha, gamma)

# Plot State Values - aggregated
plt.plot(ag.StateValues[:,0],ag.StateValues[:,1])
plt.show()

# Plot mu values (frequency of state occurances)
plt.plot(ag.mu)
plt.show()