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

    def __init__(self, no_epsiodes,layers,alpha,gamma,beta_m,beta_v,epsilon,min_epsilon,feature_ranges, number_tilings, bins, offset):

        self = Car.Car_initialise(self,layers)
        #self = Car.Adam_Parameters(self,alpha,beta_m,beta_v,epsilon)

        # Create tilings
        self.tilings = Car.create_tiling(feature_range, number_tilings, bins, offset)
        self.num_tilings = number_tilings
        self.actions = [0,1,2]
        self.lr = alpha
        self.gamma = gamma

        self.state_sizes = [tuple(len(splits) + 1 for splits in tiling) for tiling in
                            self.tilings]  # [(10, 10), (10, 10), (10, 10)]
        self.q_tables = [np.zeros(shape=(state_size + (len(self.actions),))) for state_size in self.state_sizes]
        print(self.q_tables)


        reduction = (epsilon - min_epsilon)/no_epsiodes
        
        for k in range(no_epsiodes):

            epsilon -= reduction


            if k%20 == 0:
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

            print(k)

            if k == (no_epsiodes - 10):
                print('pause')

            # Initial action randomly selected
            act = random.choice([0,1,2])
            #Car.Action_verbalise(act)
            self = Car.Cumulative_Reward(self,act,alpha,gamma,epsilon,k)

            #Visualise q_values:

            
    def create_tiling(feature_range, number_tilings, bins, offset):

        """
        feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
        number_tilings: number of tilings; example: 3 tilings
        bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
        offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
        """

        tilings = []

        for i in range(number_tilings):

            # For each tiling layer
            tile = []
            tile_bin = bins[i]
            tile_offset = offset[i]

            for j in range(len(feature_range)):
                # The bins for each feature for that tile layer
                feat_range = feature_range[j]

                feat_tile = np.linspace(feat_range[0],feat_range[1],tile_bin[j]+1)[1:-1] + tile_offset[j]
                tile.append(feat_tile)

            tilings.append(tile)

        tilings = np.array(tilings)

        return tilings

    def Car_initialise(self,layers):

        self.weights = [dict() for i in range(len(layers)-1)]
        for i in range(len(layers)-1):
            for param in ["W","b"]:
                if param == "W":
                    self.weights[i][param] = torch.zeros(layers[i],layers[i+1], device=device, dtype = dtype, requires_grad=True)
                elif param == "b":
                    self.weights[i][param] = torch.zeros(1,layers[i+1], device=device, dtype = dtype, requires_grad=True)

        return self

    def create_tiling(feature_range, number_tilings, bins, offset):

        """
        feature_ranges: range of each feature; example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
        number_tilings: number of tilings; example: 3 tilings
        bins: bin size for each tiling and dimension; example: [[10, 10], [10, 10], [10, 10]]: 3 tilings * [x_bin, y_bin]
        offsets: offset for each tiling and dimension; example: [[0, 0], [0.2, 1], [0.4, 1.5]]: 3 tilings * [x_offset, y_offset]
        """

        tilings = []

        for i in range(number_tilings):

            # For each tiling layer
            tile = []
            tile_bin = bins[i]
            tile_offset = offset[i]

            for j in range(len(feature_range)):
                # The bins for each feature for that tile layer
                feat_range = feature_range[j]

                feat_tile = np.linspace(feat_range[0],feat_range[1],int(tile_bin[j])+1)[1:-1] + tile_offset[j]
                tile.append(feat_tile)

            tilings.append(tile)

        tilings = np.array(tilings)

        return tilings

    def get_tiling(tilings,feature):

        """
        feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
        tilings: tilings with a few layers
        return: the encoding for the feature on each layer
        """

        feature_coding = []
        for tile in tilings:

            feature_tile = []

            for i in range(len(feature)):
                code = np.digitize(feature[i],tile[i])
                feature_tile.append(code)

            feature_coding.append(feature_tile)

        feature_coding = np.array(feature_coding)

        return feature_coding

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
            q_val = -10000000
            for act in [0,1,2]:

                q = Car.Action_Value(self,position,velocity,act)

                if q > q_val:
                    q_val = q
                    action = act

        return action

    def get_tile_coding(feature,tilings):

        """
        feature: sample feature with multiple dimensions that need to be encoded; example: [0.1, 2.5], [-0.3, 2.0]
        tilings: tilings with a few layers
        return: the encoding for the feature on each layer
        """

        feature_coding = []
        for tile in tilings:

            feature_tile = []

            for i in range(len(feature)):
                code = np.digitize(feature[i],tile[i])
                feature_tile.append(code)

            feature_coding.append(feature_tile)

        feature_coding = np.array(feature_coding)

        return feature_coding

    def Action_Value(self,position,velocity,act):

        state = [position,velocity]
        state_codings = Car.get_tile_coding(state,self.tilings)
        action_idx = act

        value = 0
        for coding, q_table in zip(state_codings, self.q_tables):
            # for each q table
            value += q_table[tuple(coding) + (action_idx,)]
        q_val = value/self.num_tilings

        return q_val

    def update_weight(self, prev_position, prev_velocity, previous_action, reward):
        
        state = [prev_position, prev_velocity]
        state_codings = Car.get_tile_coding(state, self.tilings)  # [[5, 1], [4, 0], [3, 0]] ...
        action_idx = previous_action

        for coding, q_table in zip(state_codings, self.q_tables):
            delta = reward + self.gamma*Car.Action_Value(self,self.CurrentPosition,self.CurrentVelocity,self.CurrentAction) - Car.Action_Value(self,prev_position,prev_velocity,previous_action)
            q_table[tuple(coding) + (action_idx,)] += self.lr * (delta)

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
        while (observation[0] < 0.49) and (t < 1200): #not done: 
            
            t += 1
            previous_Position = self.CurrentPosition
            previous_Velocity = self.CurrentVelocity
            previous_action = self.CurrentAction

            action = Car.Policy(self,previous_Position,previous_Velocity,epsilon)
            #Car.Action_verbalise(action)

            observation, reward, done, info = env.step(action)
            if observation[0]>=0.49:
                print(observation[0])
            
            # total reward
            total_reward += reward

            #env.render()
            if k > 20:
                env.render()

            self.CurrentPosition = observation[0]
            self.CurrentVelocity = observation[1]
            self.CurrentAction = action

            #reward = 0
            #print(reward)
            self = Car.update_weight(self, previous_Position, previous_Velocity, previous_action, reward)

        #print(total_reward)

        return self


no_epsiodes = 5000
epsilon = 0.1 #0.8 #0.0001
min_epsilon = 0
alpha = 0.5 #0.001
gamma = 1
no_trials = 1
beta_m = 0.9
beta_v = 0.999
#layers = [(WIN_STATE-1), 100,  1]  #N, D_in, H, D_out
layers = [3, 100,  1]

#Tiling parameters
number_tilings = 8
bin = 8
bins = np.multiply([bin,bin],np.ones([number_tilings,len(feature_range)]))
O1 = np.linspace(0,1,bin).reshape(-1,1)
O2 = np.linspace(0,0.1,bin).reshape(-1,1)
offset = np.concatenate([O1,O2],axis=1)   #[[0,0],[0.1,0.01],[0.2,0.02],[0.3,0.03],[0.4,0.04],[0.5,0.05],[0.6,0.06],[0.7,0.07]]

ag = Car(no_epsiodes,layers,alpha,gamma,beta_m,beta_v,epsilon,min_epsilon,feature_range, number_tilings, bins, offset)
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