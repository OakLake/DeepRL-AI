'''
OpenaI Gym Q-Learning Reinforcement Learning
Sammy Hasan
Feb 2018

This programme is adapted to the CartPole problem, given the different action_space and observation_space for each Gym Env.

'''

import gym
import numpy as np
import math
import Box2D



env = gym.make('CartPole-v1')



print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


H = env.observation_space.high
L = env.observation_space.low


discount = 0.99
buckets = (1,1,6,3)
q_table = np.zeros((1,1,6,3,2))


Bounds = list(zip(L, H))
Bounds[1] = (-0.5,0.5)
Bounds[3] = (-math.radians(50), math.radians(50))

def getBucket(cont):
    B = []
    for b in range(len(cont)):
        if cont[b] >= Bounds[b][1]:
            B.append(buckets[b] - 1)
        elif cont[b] <= Bounds[b][0]:
            B.append(0)
        else:
            B.append(int(  (cont[b] - Bounds[b][0]) / ((Bounds[b][1] - Bounds[b][0])/buckets[b])  ))

    return tuple(B)



for e in range(1000):
    observation = env.reset()

    alpha = max(0.1, min(0.5, 1.0 - math.log10((e+1)/25)))

    state0 = getBucket(observation)
    for t in range(200):

        env.render()

        explore = max(0.01, min(1, 1.0 - math.log10((e+1)/25)))

        if np.random.random() < explore :
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state0])

        observation, reward, done, _ = env.step(action)
        state = getBucket(observation)

        q_table[state0][action] = (1 - alpha)*q_table[state0][action] + alpha*(reward + discount*np.amax(q_table[state]) )
        state0 = state

        if done:
                print("Episode [{}] finished after [{}] timesteps :: exp [{}] alpha [{}]".format(e+1,t+1,explore,alpha))
                break
