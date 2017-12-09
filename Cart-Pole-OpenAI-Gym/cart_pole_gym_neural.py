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
        self.W0 = np.random.rand(n_struct[0],n_struct[1]) - 0.5
        self.W1 = np.random.rand(n_struct[1],n_struct[2]) - 0.5
        self.W2 = np.random.rand(n_struct[2],n_struct[3]) - 0.5
    def react(self,observation):
        in_x =  observation + [1]
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


env = gym.make('CartPole-v1')
env.reset()

best_score = 0
count = 0
Good_agents = []
for i_episode in range(10000000): # 42 trials
    agent = Agent([4,7,3,1])
    # agent.reflex_info()

    observation = env.reset()
    for t in range(501):
        # env.render()
        action = agent.react(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print ("Episode {} finished after {} timesteps".format(i_episode,t+1))
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
            print ("Finished after {} timesteps".format(t+1))
            for _ in range(10):
                env.render()
            break
