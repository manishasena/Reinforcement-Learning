
import numpy as np
import random
#import pandas as pd
import matplotlib.pyplot as plt

import torch

WIN_STATE = 501 #1001
LOSE_STATE = 0
actions = ["left","right"]
no_states = 502 #1002
START = 250 #500
jump_size = 50 #100

dtype = torch.float
device = torch.device("cpu") 

class AGENT:

    def __init__(self,epsilon, iterations, alpha, gamma):

           
        layers = [(WIN_STATE-1), 100,  1]  #N, D_in, H, D_out

        self.weights = {}
        self.bias = {}
        for i in range(len(layers)-1):
            self.weights[str(i)] = torch.randn(layers[i], layers[i+1], device=device, dtype=dtype, requires_grad=True)
            self.bias[str(i)] = torch.randn(1, layers[i+1], device=device, dtype=dtype, requires_grad=True)

        print(self.weights['0'])
        print(self.weights['1'])
        self.mu = np.zeros([int(WIN_STATE-1-LOSE_STATE),1])

        for i in range(iterations):

            if i % 1 == 0:
                print(i)
            
            self.CurrentState = START
            
            self = AGENT.Cumulative_Reward(self, gamma,alpha)

        #x = torch.tensor((np.linspace(-1,1,WIN_STATE).reshape(-1,1)), dtype=dtype)
        self.Final_State = np.zeros([1,(WIN_STATE-1)])
        for i in range((WIN_STATE-1)):
            self.Final_State[0,i] = AGENT.Value_Prediction(self,i+1)

    def one_hot_encoding(state):

        coded_vector = np.zeros([1,(WIN_STATE-1)])
        coded_vector[0,state-1] = 1

        return coded_vector

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

    def Value_Prediction(self,s):

        x = AGENT.one_hot_encoding(s)
        x = torch.tensor((x), dtype=dtype)
        v_hat = (x.mm(self.weights['0'])+self.bias['0']).clamp(min=0).mm(self.weights['1']) + self.bias['1']
        #v_hat = x.mm(self.weights['0']).clamp(min=0).mm(self.weights['1'])

        return v_hat

    def Reward(self):
        
        if self.CurrentState == WIN_STATE:
            R = 1
        elif self.CurrentState == LOSE_STATE:
            R = -1
        else:
            R = 0
            
        return R

    def mu_Update(self, state):
   
        self.mu[int(state-1)] += 1
        
        return self

    def Weight_Update(self,previous_state,R,alpha,gamma):

        #x = torch.tensor(([[previous_state/1000-0.5]]), dtype=dtype)
        #v_hat_S = (x.mm(self.weights['0'])+self.bias['0']).clamp(min=0).mm(self.weights['1']) + self.bias['1']
        v_hat_S = AGENT.Value_Prediction(self,previous_state)

        if self.CurrentState == WIN_STATE:
            v_hat_S_dash = 0
        elif self.CurrentState == LOSE_STATE:
            v_hat_S_dash = 0
        else:
            #x = torch.tensor(([[self.CurrentState/1000-0.5]]), dtype=torch.float)
            #v_hat_S_dash = (x.mm(self.weights['0'])+self.bias['0']).clamp(min=0).mm(self.weights['1']) + self.bias['1']
            v_hat_S_dash = AGENT.Value_Prediction(self,self.CurrentState)
            v_hat_S_dash = float(v_hat_S_dash.detach().numpy())

        delta = R + gamma*v_hat_S_dash - float(v_hat_S.detach().numpy())
        v_hat_S.backward()

        #print('Weight grad')
        #print(self.weights['1'].grad)

        #print('by hand weight')
        #x = AGENT.one_hot_encoding(previous_state)
        #x = torch.tensor((x), dtype=dtype)
        #h = x.mm(self.weights['0'])
        #h_relu = h.clamp(min=0)
        #y_pred = h_relu.mm(self.weights['1'])

        #grad_y_pred = 2.0 * (y_pred - (R + gamma*v_hat_S_dash))
        #grad_w2 = h_relu.t().mm(grad_y_pred)
        #print(grad_w2)
        
        mu = float(self.mu[int(previous_state-1)]/np.sum(self.mu))
        
        with torch.no_grad():
            self.weights['0'] += alpha*delta*self.weights['0'].grad
            self.weights['1'] += alpha*delta*self.weights['1'].grad
            self.bias['0'] += alpha*delta*self.bias['0'].grad
            self.bias['1'] += alpha*delta*self.bias['1'].grad

            # Manually zero the gradients after updating weights
            self.weights['0'].grad.zero_()
            self.weights['1'].grad.zero_()
            self.bias['0'].grad.zero_()
            self.bias['1'].grad.zero_()

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


iterations = 500
epsilon = 0.1
alpha = 0.05
gamma = 1
no_trials = 1

for i in range(no_trials):
    ag = AGENT(epsilon,iterations, alpha, gamma)

plt.plot(np.linspace(0,(WIN_STATE-1),(WIN_STATE-1)).reshape(-1,1),ag.Final_State.reshape(-1,1))
plt.show()
print(ag.Final_State)
print(ag.weights['0'])
print(ag.weights['1'])
print('end')