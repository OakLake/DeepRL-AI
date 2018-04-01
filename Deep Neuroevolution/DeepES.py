# Deep Neural Evolutionary Strategy
# by Sammy Hasan
# March 2018

import gym
import copy
from neuralagent import *
import os
import numpy as np

os.system('clear')
env = gym.make('Hopper-v2')


print('############################################')
print('Action Space: ', env.action_space)

print('Obsrev Space: ', env.observation_space)
print('Obsrev Space H: ', env.observation_space.high)
print('Obsrev Space L: ', env.observation_space.low)
print('############################################')

# quit()
############################################ PARAMETERS ::::

NUM_POPULATION = 100
NUM_EPISODE = 10000
Top_N = 10
MUTATION_FACTOR = 0.1
NUM_ACTIONS = 3
NUM_OBSERV = 11
# IDXES = [55,65,75,85]
NUM_AVG = 2




Population = []
for _ in range(NUM_POPULATION):
    agent = NeuralAgent([NUM_OBSERV,100,60,NUM_ACTIONS],swish)
    Population.append(agent)


for e in range(NUM_EPISODE):
    scores = [0] * NUM_POPULATION
    # Evaluating the agents
    for n_avg in range(NUM_AVG):
        # os.system('clear')
        for ix_ag, ag in enumerate(Population):
            observ = env.reset()
            ep_rwd = 0
            t = 0
            while True:
                t += 1
                action = np.argmax(ag.react(observ))#int(round(ag.react(observ)[0]))
                observ, reward, done, info = env.step(action)
                # if (ix_ag <= 10) & (e%10 == 0):
                    # env.render()

                ep_rwd += reward
                if done:
                    # print('Episode {} N_AVG {} Agent {} Reward {} Steps {}'.format(e+1,n_avg,ix_ag,ep_rwd,t))
                    scores[ix_ag] += (ep_rwd/NUM_AVG)
                    break
    print('Episode {} Population AVG acore {} '.format(e, (sum(scores)/NUM_POPULATION) ) )
    # Mutating the agents / Evolutionary Strategy done here !
    # print('Mutating')

    Pop_temp = [x for _,x in sorted(zip(scores,Population),key = lambda pair: pair[0],reverse = True)]
    Population = copy.deepcopy(Pop_temp[:Top_N])
    for ag in Pop_temp[:Top_N]:
        for _ in range(9): # 9 copies of each agent
            Population.append(ag.mutate(MUTATION_FACTOR))
    # print('Done Mutating!!')
    # for i,n in enumerate(IDXES):
        # Population[-(i+1)] = Pop_temp[n]
    # os.system('clear')













# end of programme
