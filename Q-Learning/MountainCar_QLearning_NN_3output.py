import gym
import torch
#import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm, trange

env = gym.make('MountainCar-v0')
#env.seed(1); torch.manual_seed(1); np.random.seed(1)
feature_range = [[env.min_position, env.max_position],[-1*env.max_speed,env.max_speed]]

dtype = torch.float
device = torch.device("cpu") 

loss_fn = torch.nn.MSELoss()

class Car:

    def __init__(self, no_epsiodes,layers,alpha,gamma,beta_m,beta_v,epsilon,min_epsilon):

        # Initialise NN weights
        self = Car.Car_initialise(self,layers)
        # Initialise parameters for Adam optimiser
        self = Car.Adam_Parameters(self,alpha,beta_m,beta_v,epsilon)

        # Layer information
        self.layers = layers

        # Monitor performance
        self.TD_error = list()
        self.final_position = list()
        self.success = 0 

        # Parameters
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
 
   
        for k in trange(no_epsiodes):

            # Print cost function every few episodes
            # Visualise q-values of each state, using max action
            if k%(no_epsiodes-1) == 0:
                plt.plot(self.TD_error)
                plt.ylim(0,15)
                plt.show()

                plt.plot(self.final_position)
                plt.show()

                print(k)
                x_val = np.linspace(feature_range[0][0],feature_range[0][1],50)
                y_val = np.linspace(feature_range[1][0],feature_range[1][1],50)
                X = np.zeros([len(x_val),len(y_val)])
                Y = np.zeros([len(x_val),len(y_val)])
                Z = np.zeros([len(x_val),len(y_val)])

                for l in range(len(x_val)):
                    for r in range(len(y_val)):

                        X[r,l] = x_val[l]
                        Y[r,l] = y_val[r]
                        val, _ = torch.max(Car.Action_Value(self,x_val[l],y_val[r]),-1)
                        Z[r,l] = -1*val
                        #Z[r,l] = -1*max([Car.Action_Value(self,x_val[l],y_val[r],0)[0],Car.Action_Value(self,x_val[l],y_val[r],1)[1],Car.Action_Value(self,x_val[l],y_val[r],2)[2]])

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # Plot a basic wireframe.
                ax.plot_wireframe(X, Y, Z)

                plt.show()

                print(self.success)

            self = Car.Cumulative_Reward(self,gamma,k)
            

    def Car_initialise(self,layers):

        self.weights = [dict() for i in range(len(layers)-1)]
        
        for i in range(len(layers)-1):
            self.weights[i] = {}
            for param in ["W","b"]: 
                if param == "W":
                    self.weights[i][param] = torch.randn(layers[i],layers[i+1], device=device, dtype = dtype, requires_grad=True)
                    #self.weights[i][param] = torch.nn.Linear(layers[i],layers[i+1], bias=False)
                elif param == "b":
                    self.weights[i][param] = torch.randn(1,layers[i+1], device=device, dtype = dtype, requires_grad=True)
                    #self.weights[i][param] = torch.nn.Linear(1,layers[i+1], bias=False)

        
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

        if epsilon == 0:
            _, action = torch.max(Car.Action_Value(self,position,velocity),-1)
            action = action.item()

        else:
            if np.random.rand(1) < epsilon:
                action = np.random.randint(0,3)
            else:
                _, action = torch.max(Car.Action_Value(self,position,velocity),-1)
                action = action.item()

        return action

    def Action_Value(self,position,velocity):

        x = torch.tensor([[position,velocity]], dtype = dtype)
        #x = x.reshape(-1,1)

        """
        model = torch.nn.Sequential(
            self.weights[0]["W"],
            self.weights[1]["W"]
        )
        """

        for i in range(len(self.layers)-1):

            #x = torch.mm(self.weights[i]["W"].weight,x) # + self.weights[i]['b'].weight
            x = torch.mm(x,self.weights[i]["W"])
            #x = torch.mm(x,self.weights[i]["W"]) + self.weights[i]['b']

            #if i != (len(self.layers)-2):
            #    x = x.clamp(min=0) #ReLU
            #    x = torch.sign(x)
        
        #q_val = model(x)
        q_val = x.reshape(1,-1)

        return q_val

    def SGD_Weight_Update(self,next_Position,next_Velocity,next_action,done,reward,alpha):

        ############### Update weights

        q_hat_S_a = Car.Action_Value(self,self.CurrentPosition,self.CurrentVelocity)
        q_hat_S_a = q_hat_S_a[0]

        # Value estimate for current state
        q_hat_S_a_dash = Car.Action_Value(self,next_Position,next_Velocity)
        q_hat_S_a_dash = q_hat_S_a_dash[0]

        Q_target = q_hat_S_a.clone()
        Q_target = Q_target.data
        Q_target[self.CurrentAction] = reward + torch.mul(gamma,q_hat_S_a_dash[next_action].detach())
        Q_target = Q_target.reshape(q_hat_S_a.shape) 
    
        loss = loss_fn(q_hat_S_a,Q_target)
        loss.backward()

        self.TD_error.append(loss.item())
    
        with torch.no_grad():
            for i in range(len(self.layers)-1):
                #self.weights[i]['W'].weight -= self.alpha*self.weights[i]['W'].weight.grad
                self.weights[i]['W'] -= self.alpha*self.weights[i]['W'].grad
                
                #self.weights[i]['b'].weight -= self.alpha*self.weights[i]['b'].weight.grad
                #self.weights[i]['b'] -= self.alpha*self.weights[i]['b'].grad

                # Manually zero the gradients after updating weights
                #self.weights[i]['W'].weight.grad.zero_()
                self.weights[i]['W'].grad.zero_()
                
                #self.weights[i]['b'].weight.grad.zero_()
                #self.weights[i]['b'].grad.zero_()

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

        q_hat_S_a.backward()
        delta = reward + self.gamma*q_hat_S_a_dash - float(q_hat_S_a.detach().numpy())
        self.TD_error.append(delta)
        

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

    def Cumulative_Reward(self,gamma,k):
        
        observation = env.reset()

        # Initial action randomly selected
        #act = random.choice([0,1,2])
        #observation, reward, done, info = env.step(act) # take a random action

        # total reward
        total_reward = 0

        max_position = -0.7

        for s in range(2000): #not done: 

            self.CurrentPosition = observation[0]
            self.CurrentVelocity = observation[1]

            # Given S, select A
            action = Car.Policy(self,self.CurrentPosition,self.CurrentVelocity,self.epsilon)
            self.CurrentAction = action

            #print(observation)
            # Take action, and return reward and S'
            observation, reward, done, info = env.step(action)

            if k > 2800:
                env.render()

            if observation[0] > max_position:
                max_position = observation[0]

            next_Position = observation[0]
            next_Velocity = observation[1]


            ### Q Learning Option ###
            # Given S', select next A
            action = Car.Policy(self,next_Position,next_Velocity,0)
            next_action = action
            
            #observation, reward, done, info = env.step(action)

            if observation[0] >= 0.5:
                #print(k)
                print('Reached!')
                self.success += 1
                print(self.success)
                self.alpha = self.alpha*0.9
                self.epsilon = self.epsilon*0.99
            
            # total reward
            total_reward += reward

            if done:
                break
            ########################

            #self = Car.Adam_Weight_Update(self,previous_Position,previous_Velocity,previous_action,done,reward,alpha,epsilon)
            self = Car.SGD_Weight_Update(self,next_Position,next_Velocity,next_action,done,reward,alpha)

        self.final_position.append(max_position)

        return self


no_epsiodes = 3000
epsilon = 0.3 #0.8 #0.0001
min_epsilon = 0.25
alpha = 0.001 #0.001
gamma = 0.99
observation = env.reset()
success = 0

no_trials = 1
beta_m = 0.9
beta_v = 0.999
#layers = [(WIN_STATE-1), 100,  1]  #N, D_in, H, D_out
#layers = [3, 1000,  1]
layers = [2, 200, 3]


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
