import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

env = gym.make('MountainCar-v0')
feature_range = [[env.min_position, env.max_position],[-1*env.max_speed,env.max_speed]]

dtype = torch.float
device = torch.device("cpu") 


class Car:

    def __init__(self, no_epsiodes,layers,alpha,gamma,beta_m,beta_v,epsilon,min_epsilon):

        # Initialise NN weights
        self = Car.Car_initialise(self,layers)
        # Initialise parameters for Adam optimiser
        self = Car.Adam_Parameters(self,alpha,beta_m,beta_v,epsilon)

        self.layers = layers

        self.TD_error = list()
        self.final_position = list()

        self.gamma = gamma

        reduction = (epsilon - min_epsilon)/no_epsiodes
        
        for k in range(no_epsiodes):

            #print(k)

            # Reduce exploration in each step
            epsilon -= reduction

            # Print cost function every few episodes
            # Visualise q-values of each state, using max action
            if k%100 == 0:
                plt.plot(self.TD_error)
                plt.ylim(-5,5)
                plt.show()

                plt.plot(self.final_position)
                plt.show()

                print(k)
                x_val = np.linspace(feature_range[0][0],feature_range[0][1],100)
                y_val = np.linspace(feature_range[1][0],feature_range[1][1],100)
                X = np.zeros([len(x_val),len(y_val)])
                Y = np.zeros([len(x_val),len(y_val)])
                Z = np.zeros([len(x_val),len(y_val)])

                for l in range(len(x_val)):
                    for r in range(len(y_val)):

                        X[r,l] = x_val[l]
                        Y[r,l] = y_val[r]
                        Z[r,l] = -1*max([Car.Action_Value(self,x_val[l],y_val[r],0),Car.Action_Value(self,x_val[l],y_val[r],1),Car.Action_Value(self,x_val[l],y_val[r],2)])

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # Plot a basic wireframe.
                ax.plot_wireframe(X, Y, Z)

                plt.show()

            # Initial action randomly selected
            act = random.choice([0,1,2])
            #Car.Action_verbalise(act)
            self = Car.Cumulative_Reward(self,act,alpha,gamma,epsilon,k)
            

    def Car_initialise(self,layers):

        self.weights = [dict() for i in range(len([0,1,2]))]
        #self.weights = [dict() for i in range(len(layers)-1)]

        for act in [0,1,2]:
            for i in range(len(layers)-1):
                self.weights[act][i] = {}
                for param in ["W","b"]:
                    if param == "W":
                        #self.weights[act][i][param] = torch.zeros(layers[i],layers[i+1], device=device, dtype = dtype, requires_grad=True)
                        self.weights[act][i][param] = torch.rand(layers[i],layers[i+1], device=device, dtype = dtype, requires_grad=True)
                    elif param == "b":
                        #self.weights[i][param] = torch.zeros(1,layers[i+1], device=device, dtype = dtype, requires_grad=True)
                        self.weights[act][i][param] = torch.randn(1,layers[i+1], device=device, dtype = dtype, requires_grad=True)

        return self

    def Adam_Parameters(self,alpha,beta_m,beta_v,epsilon):

        self.step_size = alpha
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.epsilon = epsilon

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(len(layers)-1)]
        self.v = [dict() for i in range(len(layers)-1)]

        for i in range(len(layers)-1):
            # Initialize Adam parameter values
            self.m[i]["W"] = np.zeros((layers[i], layers[i+1]))
            self.m[i]["b"] = np.zeros((1, layers[i+1]))
            self.v[i]["W"] = np.zeros((layers[i], layers[i+1]))
            self.v[i]["b"] = np.zeros((1, layers[i+1]))

        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

        return self

    def Action_verbalise(action_no):
        
        action_dic = {}
        action_dic[0] = 'left'
        action_dic[1] = 'stationary'
        action_dic[2] = 'right'

        verbal_action = action_dic[action_no]
        print(verbal_action)
        return verbal_action

    def Policy(self,position,velocity,epsilon):

        #Select which next action to take. 
        #This is defined as the action with the highest state value

        if np.random.uniform() <= epsilon:
            action = random.choice([0,1,2])
        else:

            q0 = Car.Action_Value(self,position,velocity,0)
            q1 = Car.Action_Value(self,position,velocity,1)
            q2 = Car.Action_Value(self,position,velocity,2)

            action = np.argmax([q0,q1,q2])

        return action

    def Action_Value(self,position,velocity,act):

        # Normalise inputs to NN
        #position = (position + 1.2)/(0.5+1.2)
        #velocity = (velocity +0.07)/(0.07 + 0.07)
        #act = act/2

        #x = torch.tensor([[position,velocity,act]], dtype = dtype)
        x = torch.tensor([[position,velocity]], dtype = dtype)

        for i in range(len(self.layers)-1):
            x = (x.mm(self.weights[act][i]["W"])+self.weights[act][i]['b'])
            #x = x.mm(self.weights[act][i]["W"])
            #x = torch.sigmoid(x)
            x = x.clamp(max=0) #ReLU
            
            
        q_val = x

        return q_val

    def SGD_Weight_Update(self,previous_position,previous_velocity,prev_action,done,R,alpha):

        # Value estimate for previous state
        #v_hat_S = AGENT.Value_Prediction(self,previous_state)
        q_hat_S_a = Car.Action_Value(self,previous_position,previous_velocity,prev_action)

        # Value estimate for current state
        if self.CurrentPosition >= 0.5:
            q_hat_S_a_dash = 0
        else:
            #v_hat_S_dash = AGENT.Value_Prediction(self,self.CurrentState)
            #v_hat_S_dash = float(v_hat_S_dash.detach().numpy())
            q_hat_S_a_dash = Car.Action_Value(self,self.CurrentPosition,self.CurrentVelocity,self.CurrentAction)
            q_hat_S_a_dash = float(q_hat_S_a_dash.detach().numpy())

        q_hat_S_a.backward()
        delta = R + self.gamma*q_hat_S_a_dash - float(q_hat_S_a.detach().numpy())
        #delta =-1*(R + self.gamma*q_hat_S_a_dash - q_hat_S_a)
        self.TD_error.append(delta)
        #delta.backward()
        
        
        with torch.no_grad():
            for i in range(len(self.layers)-1):
                self.weights[prev_action][i]['W'] += alpha*delta*self.weights[prev_action][i]['W'].grad
                self.weights[prev_action][i]['b'] += alpha*delta*self.weights[prev_action][i]['b'].grad

                # Manually zero the gradients after updating weights
                self.weights[prev_action][i]['W'].grad.zero_()
                self.weights[prev_action][i]['b'].grad.zero_()

        return self

    def Adam_Weight_Update(self,previous_position,previous_velocity,prev_action,done,reward,alpha,epsilon):

        """
        #Given weights and update g, return updated weights
        """

        q_hat_S_a = Car.Action_Value(self,previous_position,previous_velocity,prev_action)

        if self.CurrentPosition >= 0.5:
            q_hat_S_a_dash = 0
        else:
            q_hat_S_a_dash = Car.Action_Value(self,self.CurrentPosition,self.CurrentVelocity,self.CurrentAction)
            q_hat_S_a_dash = float(q_hat_S_a_dash.detach().numpy())

        delta = reward + self.gamma*q_hat_S_a_dash - float(q_hat_S_a.detach().numpy())
        self.TD_error.append(delta)
        q_hat_S_a.backward()

        # Get gradient of each weight
        g = [dict() for i in range(len(self.weights[0]))]
        with torch.no_grad():
            for i in range(len(self.weights[0])):
                for param in self.weights[i].keys():

                    g[i][param] = delta*self.weights[i][param].grad

                    self.weights[i][param].grad.zero_()
        
        for i in range(len(self.weights[0])):
            for param in self.weights[i].keys():

                ### update self.m and self.v
                self.m[i][param] = self.beta_m * self.m[i][param] + (1-self.beta_m) * g[i][param].detach().numpy()
                self.v[i][param] = self.beta_v * self.v[i][param] + (1-self.beta_v) * (g[i][param].detach().numpy() * g[i][param].detach().numpy())

                ### compute m_hat and v_hat
                m_hat = self.m[i][param]/(1-self.beta_m_product)
                v_hat = self.v[i][param]/(1-self.beta_v_product)

                ### update weights
                x = (self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon))
                self.weights[i][param] = torch.tensor((self.weights[i][param].detach().numpy()) + x, device=device, dtype=dtype, requires_grad=True)
                
        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return self

    def Cumulative_Reward(self,act,alpha,gamma,epsilon,k):
        env.reset()
        observation, reward, done, info = env.step(act) # take a random action

        # total reward
        total_reward = reward

        self.CurrentPosition = observation[0]
        self.CurrentVelocity = observation[1]
        self.CurrentAction = act

        t = 0
        while (observation[0] < 0.5) and (t<600): #not done: 
            
            t += 1
            
            previous_Position = self.CurrentPosition
            previous_Velocity = self.CurrentVelocity
            previous_action = self.CurrentAction

            action = Car.Policy(self,previous_Position,previous_Velocity,epsilon)

            observation, reward, done, info = env.step(action)

            if observation[0] >= 0.5:
                print('Reached!')
            
            # total reward
            total_reward += reward

            #env.render()

            self.CurrentPosition = observation[0]
            self.CurrentVelocity = observation[1]
            self.CurrentAction = action

            #self = Car.Adam_Weight_Update(self,previous_Position,previous_Velocity,previous_action,done,reward,alpha,epsilon)
            self = Car.SGD_Weight_Update(self,previous_Position,previous_Velocity,previous_action,done,reward,alpha)

        self.final_position.append(observation[0])

        return self


no_epsiodes = 5000
epsilon = 0.3 #0.8 #0.0001
min_epsilon = 0
alpha = 0.01 #0.001
gamma = 0.90

no_trials = 1
beta_m = 0.8
beta_v = 0.9
#layers = [(WIN_STATE-1), 100,  1]  #N, D_in, H, D_out
#layers = [3, 1000,  1]
layers = [2, 10, 20, 10, 1]


ag = Car(no_epsiodes,layers,alpha,gamma,beta_m,beta_v,epsilon,min_epsilon)
env.close()

""" for _ in range(no_epsiodes):
    done = False
    env.reset()
    while not done:
        env.render()

        # action = 0: left, action = 1: stationary, action = 2: right
        action = env.action_space.sample()  #take a random action

        observation, reward, done, info = env.step(action) 
        print('Position: ' + str(observation[0]) + ' Velocity: ' + str(observation[1]))

        # When to terminate epsiode:
        if done:
            print(done)
            print('Episode finished')
            #break
env.close() """
