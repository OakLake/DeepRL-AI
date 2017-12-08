# Testing the OpenAI gym environment with the cart pole system
# Sammy Hasan
# 2017

import gym
import random

class Agent():
    '''Agent with simple proportional response'''
    def __init__(self,inf_sub):
        self.a = random.uniform(-inf_sub,inf_sub)
        self.b = random.uniform(-inf_sub,inf_sub) # specfic to CartPole env
        self.c = random.uniform(-inf_sub,inf_sub)
        self.d = random.uniform(-inf_sub,inf_sub) # specfic to CartPole env
    def react(self,observation):
        reaction = self.a*observation[0] + self.b*observation[1] + self.c*observation[2] + self.d*observation[3]
        return reaction > 0 # cart env has a discrete space if (2) pushing [0,1]:[left,righ]
    def reflex_info(self):
        return "[{}, {}, {}, {}]".format(self.a,self.b,self.c,self.d)


env = gym.make('CartPole-v1')
env.reset()

best_score = 0
for i_episode in range(100): # 42 trials
    agent = Agent(30)
    reflexes = agent.reflex_info()
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = agent.react(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print ("Episode {} finished after {} timesteps :: {}".format(i_episode,t+1,reflexes))
            if (t+1) > best_score:
                best_score = t+1
            break

print("Best Score: ", best_score)
