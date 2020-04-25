
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
no_aggregation = 100

class AGENT:

    def __init__(self,epsilon, iterations, alpha, gamma):
        
        self.weights = np.zeros([int((WIN_STATE-1-LOSE_STATE)//no_aggregation),1])
        
        self.mu = np.zeros([int((WIN_STATE-1-LOSE_STATE)//no_aggregation),1])

        self.error = np.zeros([iterations,2])
        TrueStateValue = np.linspace(-1,1,int((WIN_STATE-1-LOSE_STATE)//no_aggregation)).reshape(-1,1)

        for i in range(iterations):
            
            self.CurrentState = START
            
            self = AGENT.Cumulative_Reward(self, gamma,alpha)

            # Error after each episode
            self.error[i,0] = i+1
            self.error[i,1] = np.sum((self.mu/(np.sum(self.mu)))*(TrueStateValue-self.weights)**2)

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

    def mu_Update(self, state):
   
        self.mu[int((state-1)//no_aggregation)] += 1
        
        return self

    def Weight_Update(self,previous_state,R,alpha,gamma):

        x_S = np.zeros([int((WIN_STATE-1-LOSE_STATE)//no_aggregation),1]) 
        x_S[int((previous_state-1)//no_aggregation)] = 1

        v_hat_S = np.sum(self.weights*x_S)

        if self.CurrentState == WIN_STATE:
            v_hat_S_dash = 0
        elif self.CurrentState == LOSE_STATE:
            v_hat_S_dash = 0
        else:
            x_S_dash = np.zeros([int((WIN_STATE-1-LOSE_STATE)//no_aggregation),1]) 
            x_S_dash[int((self.CurrentState-1)//no_aggregation)] = 1
            v_hat_S_dash = np.sum(self.weights*x_S_dash)
        
        mu = self.mu[int((previous_state-1)//no_aggregation)]/np.sum(self.mu)
        
        self.weights = self.weights + mu*alpha*(R + gamma*v_hat_S_dash - v_hat_S)*x_S

        return self

    def Cumulative_Reward(self, gamma,alpha):
        
        while (self.CurrentState != WIN_STATE) and (self.CurrentState != LOSE_STATE):
            
                # Add state to list
                previous_state = self.CurrentState

                self = AGENT.mu_Update(self, previous_state)
                
                act, hops = AGENT.Policy(self,epsilon)

                self = AGENT.Next_State(self,act,hops)
                R = AGENT.Reward(self)

                self = AGENT.Weight_Update(self,previous_state,R,alpha,gamma)
        
        return self


iterations = 50
epsilon = 0.1
alpha = 0.22
gamma = 0.9
no_trials = 3

n_holder = np.zeros([no_trials,iterations])

for i in range(no_trials):
    ag = AGENT(epsilon,iterations, alpha, gamma)
    n_holder[i,:] = ag.error[:,1]


# Plot State Values - aggregated
plt.plot(ag.error[:,0], np.mean(n_holder,axis = 0))
plt.title('Semi-Gradient TD with State Aggregation - ERROR')
plt.xlabel('Episodes')
plt.ylabel('Mean Squared Error')
#plt.ylim(-1,1)
plt.show()

# Plot State Values - aggregated
plt.plot(ag.StateValues[:,0],ag.StateValues[:,1])
plt.title('Semi-Gradient TD with State Aggregation')
plt.xlabel('State')
plt.ylabel('State Value')
plt.ylim(-1,1)
plt.show()

