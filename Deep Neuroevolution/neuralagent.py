import copy
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def swish(x):
    return x*sigmoid(x)
def softmax(x):
    return np.exp(x)/sum(np.exp(x))

class NeuralAgent():
    '''Agent with fixed neural network stucture, with swish activation function then sigmoid'''
    def __init__(self,n_struct,func):
        self.activation_func = func
        self.nn_struct = n_struct
        self.W0 = np.random.rand(self.nn_struct[0],self.nn_struct[1]) - 0.5
        self.W1 = np.random.rand(self.nn_struct[1],self.nn_struct[2]) - 0.5
        self.W2 = np.random.rand(self.nn_struct[2],self.nn_struct[3]) - 0.5
        self.B1 = [0]*(self.nn_struct[1])
        self.B2 = [0]*(self.nn_struct[2])
        self.B3 = [0]*(self.nn_struct[3])
        self.W0 /= np.sum(self.W0)
        self.W1 /= np.sum(self.W1)
        self.W2 /= np.sum(self.W2)

    def react(self,observation):

        in_x = np.array(observation)
        y = np.matmul(in_x,self.W0) + self.B1
        sw = swish(y)
        y = np.matmul(sw,self.W1) + self.B2
        sw = swish(y)
        y = np.matmul(sw,self.W2) + self.B3
        # A = softmax(y)

        return y

    def mutate(self,factor):
        mutant = NeuralAgent(self.nn_struct,self.activation_func)
        mutant.W0 = self.W0 + (np.random.rand(self.nn_struct[0],self.nn_struct[1]) - 0.5) * factor
        mutant.W1 = self.W1 + (np.random.rand(self.nn_struct[1],self.nn_struct[2]) - 0.5) * factor
        mutant.W2 = self.W2 + (np.random.rand(self.nn_struct[2],self.nn_struct[3]) - 0.5) * factor
        mutant.B1 = self.B1 + (np.random.rand(self.nn_struct[1]) - 0.5) * factor
        mutant.B2 = self.B2 + (np.random.rand(self.nn_struct[2]) - 0.5) * factor
        mutant.B3 = self.B3 + (np.random.rand(self.nn_struct[3]) - 0.5) * factor
        return mutant

    def reflex_info(self):
        print("W0: {} \nW1: {}".format(self.W0,self.W1))

    def nn(self):
        return '5x{}x{}x1'.format(self.nn_struct[1],self.nn_struct[2])
