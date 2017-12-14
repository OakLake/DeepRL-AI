# Q Learning for OpenAI Gym
# Sammy Hasan
# 2017

import gym
import numpy as np
import sys,os
import pickle


env = gym.make('MountainCar-v0')
env.reset()

print('...')
print('Action Space: ',env.action_space)
print('Obsrv  Space: ',env.observation_space)
print('Obsrv High:',env.observation_space.high)
print('Obsrv Low :',env.observation_space.low)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def swish(x):
    return x*sigmoid(x)
def softmax(x):
    e_x = np.exp(x)
    sm = e_x/np.sum(e_x)
    return sm

class Agent():
    '''Agent with fixed neural network stucture, with swish activation function then sigmoid'''
    def __init__(self,n_struct):
        self.nn_struct = n_struct
        self.W0 = np.random.rand(n_struct[0],n_struct[1]) - 0.5
        self.W1 = np.random.rand(n_struct[1],n_struct[2]) - 0.5
        self.W2 = np.random.rand(n_struct[2],n_struct[3]) - 0.5
        self.W3 = np.random.rand(n_struct[3],n_struct[4]) - 0.5
    def react(self,observation):

        in_x = np.array([observation[0],observation[1],1.0])
        y = np.matmul(in_x,self.W0)
        sw = swish(y)
        y = np.matmul(sw,self.W1)
        sw = swish(y)
        y = np.matmul(sw,self.W2)
        sw = swish(y)
        y = np.matmul(sw,self.W3)
        A = sigmoid(y)
        A = softmax(A)
        reaction = np.argmax(A)
        return reaction # cart env has a discrete space if (2) pushing [0,1]:[left,righ]
    def get_genome(self):
        w0_re = self.W0.reshape(-1)
        w1_re = self.W1.reshape(-1)
        w2_re = self.W2.reshape(-1)
        w3_re = self.W3.reshape(-1)
        genome = np.concatenate([w0_re,w1_re,w2_re,w3_re])
        return genome
    def set_weights(self,genome):
        a = self.nn_struct[0]
        b = self.nn_struct[1]
        c = self.nn_struct[2]
        d = self.nn_struct[3]
        e = self.nn_struct[4]
        self.W0 = genome[0:a*b].reshape(a,b)
        self.W1 = genome[0:b*c].reshape(b,c)
        self.W2 = genome[0:c*d].reshape(c,d)
        self.W3 = genome[0:d*e].reshape(d,e)

    def reflex_info(self):
        print("W0: {} \nW1: {}\n W2: {}\n W3: {}".format(self.W0,self.W1,self.W2,self.W3))
    def nn(self):
        return '3x{}x{}x{}x3'.format(self.nn_struct[1],self.nn_struct[2],self.nn_struct[3])

def mate(A,B,times):
    genome_A = A.get_genome()
    genome_B = B.get_genome()

    genome_child_A = np.copy(genome_A)
    genome_child_B = np.copy(genome_B) # same as genome_A

    # swapping single weight
    for _ in range(times):
        r_ix = np.random.randint(0,genome_A.shape[0])
        genome_child_A[r_ix] = genome_B[r_ix]
        genome_child_B[r_ix] = genome_A[r_ix]

    # for gene_ix in range(genome_A.shape[0]):
    #     if np.random.rand() >= 0.75:
    #         genome_child_A[gene_ix] = genome_A[gene_ix]
    #         genome_child_B[gene_ix] = genome_B[gene_ix]
    #     else:
    #         genome_child_A[gene_ix] = genome_B[gene_ix]
    #         genome_child_B[gene_ix] = genome_A[gene_ix]

    return genome_child_A,genome_child_B

def test_agent(agent,render = False):
    observation = env.reset()
    rtn = 0
    for t in range(200):
        if render:
            env.render()
        action = agent.react(observation)
        observation, reward, done, info = env.step(action)
        rtn += reward
        if done:
            break
    return rtn

def obsrv_to_state(obs,bins):
    H = env.observation_space.high
    L = env.observation_space.low
    de = (H - L)/bins # steps
    a = int((obs[0] - L[0])/de[0])
    b = int((obs[1] - L[1])/de[1])
    return a,b

def generate_neural_population(popSize,maxItr,fill=True):
    max_rtn = -200
    e = 0
    ag_count = 0
    pop = []
    while( (len(Population) < popSize) & (e < maxItr) ):

        rtn = 0
        agent = Agent([3,10,7,4,3])
        observation = env.reset()
        for t in range(200):
            action = agent.react(observation)
            observation, reward, done, info = env.step(action)
            rtn += reward
            if done:
                if e%1000 == 0:
                    print('E: {} Done after {} steps :: Return: {}'.format(e,t,rtn))
                if rtn > max_rtn:
                    # max_rtn = rtn
                    ag_count += 1
                    Population.append([rtn,agent])
                    print('### <REWARD>: ',rtn, 'Agents so far: ',ag_count)
                else:
                    del agent
                break
        e +=1

    if (len(pop) < popSize) & fill:
        temp = pop[:]
        for _ in range(int(popSize/len(pop))):
            pop = pop + temp
        pop = pop[:popSize]

    return pop


#Population = generate_neural_population(100,int(1e7))



## Pickle
# print('Pickling Agents')
# with open('./pickled_agents.pickle','wb') as f:
#     pickle.dump(Population,f,protocol=pickle.HIGHEST_PROTOCOL)
# print('Done Pickling')

print('Loading Pickled Agents')
with open('./pickled_agents.pickle','rb') as f:
    Population = pickle.load(f)
print('Done Loading')

print('Sorting Population by score')
Population.sort(key = lambda Population: Population[0],reverse = True)

agent = Population[0][1]
_ = test_agent(agent,render=True)









##
