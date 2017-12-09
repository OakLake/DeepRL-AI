# Testing the OpenAI gym environment with the cart pole system
# Sammy Hasan
# 2017

import gym, random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def swish(x):
    return x*sigmoid(x)

class Agent():
    '''Agent with fixed neural network stucture, with swish activation function then sigmoid'''
    def __init__(self,n_struct):
        self.nn_struct = n_struct
        self.W0 = np.random.rand(n_struct[0],n_struct[1]) - 0.5
        self.W1 = np.random.rand(n_struct[1],n_struct[2]) - 0.5
        self.W2 = np.random.rand(n_struct[2],n_struct[3]) - 0.5
    def react(self,observation):

        in_x = np.array([observation[0],observation[1],observation[2],observation[3],1.0])
        y = np.matmul(in_x,self.W0)
        sw = swish(y)
        y = np.matmul(sw,self.W1)
        sw = swish(y)
        y = np.matmul(sw,self.W2)
        A = sigmoid(y)
        reaction = int(A > 0.5)
        return reaction # cart env has a discrete space if (2) pushing [0,1]:[left,righ]
    def reflex_info(self):
        print("W0: {} \nW1: {}".format(self.W0,self.W1))
    def nn(self):
        return '5x{}x{}x1'.format(self.nn_struct[1],self.nn_struct[2])


env = gym.make('CartPole-v1')
env.reset()

best_score = 100
count = 0
Good_agents = []

simL = int(1e5)
for i_episode in range(simL): # 42 trials
    H1 = np.random.randint(5,12)
    H2 = np.random.randint(2,5)
    agent = Agent([5,H1,H2,1])
    # agent.reflex_info()

    observation = env.reset()
    for t in range(501):
        # env.render()
        action = agent.react(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print ("Episode {} finished after {} timesteps ::: {}% ".format(i_episode,t+1,100*i_episode/simL))
            if (t+1) > best_score:
                count += 1
                best_score = t+1
                Good_agents.append(agent)
            else:
                del agent
            break

print("Best Score: {}, Count: {}".format(best_score,count))

_ = input('PRESS ENTER TO SEE BEST PERFORMERS')
for ag in Good_agents:
    observation = env.reset()
    for t in range(501):
        env.render()
        action = ag.react(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print ("Finished after {} timesteps :: Agent NN: {}".format((t+1),ag.nn()))
            for _ in range(20):
                env.render()
            break
