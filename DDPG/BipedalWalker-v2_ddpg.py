'''
''
'
    Deep Deterministic Policy Gradient :: https://arxiv.org/abs/1509.02971
'
''
TODO:
1 - HER
2 - Correctness of classes and learning
'''

import tensorflow as tf
import gym
import random
import numpy as np
import os


os.system('clear')


env_name = 'BipedalWalker-v2'
env = gym.make(env_name)
print('############################################')
print('Action Space: ', env.action_space)
print('Obsrev Space: ', env.observation_space)
print('Obsrev H: ',env.observation_space.high)
print('Obsrev L: ',env.observation_space.low)
print('############################################')


############################################ Algorithm Parameters ############################################

NUM_INPUTS    = 24   # state + goal array
NUM_ACTION    = 4   # number of actions

NUM_HIDDEN1   = 400
NUM_HIDDEN2   = 300
GAMMA         = 0.99
CRITIC_ALPHA  = 1e-3
ACTOR_ALPHA   = 1e-4
ACTION_BOUNDS = np.array([4,4,4,4],dtype=np.float32)
NUM_EPISODES  = 1000
MEMORY_SIZE   = 5000
TAU           = 0.01
MINIBATCH_SIZE= 32
RENDER_SKIP   = 10
OrnUhl_SIGMA  = 0.2
OrnUhl_MEAN   = np.zeros((NUM_ACTION,1))
FINAL_ACTOR_RNGE  = 3e-3
FINAL_CRITIC_RNGE = 3e-3
l2_LOSS_FACTOR = 1e-2


############################################ Critic Network ############################################
class CriticNet(object):
        def __init__(self,sess,learning_rate):
            self.sess = sess
            self.learning_rate = learning_rate
            # self.graph = tf.Graph()

            with tf.name_scope('Critic_Parameters'):
                self.w1 = tf.Variable(tf.random_uniform((NUM_INPUTS,NUM_HIDDEN1),minval = - 1/np.sqrt(NUM_INPUTS),maxval = 1/np.sqrt(NUM_INPUTS)))
                self.b1 = tf.Variable(tf.constant(0.01,shape =[NUM_HIDDEN1]))
                self.w2 = tf.Variable(tf.random_uniform(shape=(NUM_HIDDEN1,NUM_HIDDEN2),minval = - 1/np.sqrt(NUM_HIDDEN1),maxval = 1/np.sqrt(NUM_HIDDEN1)))
                self.b2 = tf.Variable(tf.constant(0.01,shape =[NUM_HIDDEN2]))
                self.w2A = tf.Variable(tf.random_uniform(shape=(NUM_ACTION,NUM_HIDDEN2),minval = - 1/np.sqrt(NUM_ACTION),maxval = 1/np.sqrt(NUM_ACTION)))
                self.w3 = tf.Variable(tf.random_uniform(shape=(NUM_HIDDEN2,1),minval = - FINAL_CRITIC_RNGE,maxval = FINAL_CRITIC_RNGE))
                self.b3 = tf.Variable(tf.constant(0.01,shape = [1]))


            with tf.name_scope('Critic_Inference'):
                self.tf_state_input = tf.placeholder(tf.float32,shape=(None,NUM_INPUTS))
                self.tf_action_input = tf.placeholder(tf.float32,shape=(None,NUM_ACTION))
                self.tf_target = tf.placeholder(tf.float32,shape=(None))

                self.action_input = self.tf_action_input # TODO remove redundancy

                z1 = tf.matmul(self.tf_state_input,self.w1) + self.b1
                a1 = tf.nn.relu(z1)
                z2 = tf.matmul(a1,self.w2)
                zA = tf.matmul(self.tf_action_input,self.w2A)
                z  = z2 + zA + self.b2
                a2 = tf.nn.relu(z)
                self.y_ = tf.matmul(a2,self.w3) + self.b3 # Q(s,a)

            with tf.name_scope('Critic_Gradients'):
                self.gradients =  tf.gradients(self.y_,self.action_input)

            with tf.name_scope('Critic_Optimize'):
                self.loss = tf.reduce_mean(tf.square(self.tf_target-self.y_)) + l2_LOSS_FACTOR*tf.nn.l2_loss(self.w1) + l2_LOSS_FACTOR*tf.nn.l2_loss(self.w2) \
                + l2_LOSS_FACTOR*tf.nn.l2_loss(self.w2A) +  l2_LOSS_FACTOR*tf.nn.l2_loss(self.w3)
                self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


        def predict(self,states_batch,actions_batch):
            y_ = self.sess.run([self.y_],feed_dict={self.tf_state_input:states_batch,self.tf_action_input:actions_batch})
            return y_

        def train(self,states_batch,actions_batch,target_batch):
            self.sess.run(self.optimize,feed_dict={self.tf_state_input:states_batch,self.tf_action_input:actions_batch,self.tf_target:target_batch})

        def get_gradients(self,states,actions):
            gradients = self.sess.run([self.gradients],feed_dict={self.tf_state_input:states,self.tf_action_input:actions})
            return gradients

        def get_parameters(self):
            tw1,tw2,tw2A,tw3,tb1,tb2,tb3 = self.sess.run([self.w1,self.w2,self.w2A,self.w3,self.b1,self.b2,self.b3])
            return (tw1,tw2,tw2A,tw3,tb1,tb2,tb3)


############################################ Actor Network ############################################
class ActorNet(object):
    def __init__(self,sess,learning_rate,action_bounds):
        self.sess = sess
        self.learning_rate = learning_rate
        self.action_bounds = action_bounds
        # self.graph = tf.Graph()

        with tf.name_scope('Actor_Parameters'):
            self.w1 = tf.Variable(tf.random_uniform(shape=(NUM_INPUTS,NUM_HIDDEN1),minval = - 1/np.sqrt(NUM_INPUTS),maxval = 1/np.sqrt(NUM_INPUTS)))
            self.b1 = tf.Variable(tf.constant(0.01,shape =[NUM_HIDDEN1]))
            self.w2 = tf.Variable(tf.random_uniform(shape=(NUM_HIDDEN1,NUM_HIDDEN2),minval = - 1/np.sqrt(NUM_HIDDEN1),maxval = 1/np.sqrt(NUM_HIDDEN1)))
            self.b2 = tf.Variable(tf.constant(0.01,shape =[NUM_HIDDEN2]))
            self.w3 = tf.Variable(tf.random_uniform(shape=(NUM_HIDDEN2,NUM_ACTION),minval = - FINAL_ACTOR_RNGE,maxval = FINAL_ACTOR_RNGE))
            self.b3 = tf.Variable(tf.constant(0.01,shape = [NUM_ACTION]))


        with tf.name_scope('Actor_Inference'):
            self.tf_state_goal_input = tf.placeholder(tf.float32,shape=(None,NUM_INPUTS))

            z1 = tf.matmul(self.tf_state_goal_input,self.w1) + self.b1
            a1 = tf.nn.relu(z1)
            z2 = tf.matmul(a1,self.w2) + self.b2
            a2 = tf.nn.relu(z2)
            z3 = tf.matmul(a2,self.w3) + self.b3
            a3 = tf.nn.tanh(z3)
            self.y_ = tf.multiply(a3,self.action_bounds)

        with tf.name_scope('Actor_Gradients'):
            self.tf_critic_gradients = tf.placeholder(tf.float32,shape=[None,NUM_ACTION])
            self.params = [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]#tf.trainable_variables(scope='Actor_Parameters') #
            self.raw_gradients = tf.gradients(self.y_,self.params, -self.tf_critic_gradients)
            self.gradients_ = list(map(lambda x : tf.divide(x,MINIBATCH_SIZE) , self.raw_gradients))


        with tf.name_scope('Actor_Optimize'):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.gradients_,self.params))
            # ^ order of self.gradients,self.params is important


    def predict(self,states_goal_batch):
        y_ = self.sess.run([self.y_],feed_dict={self.tf_state_goal_input:states_goal_batch})
        return y_

    def train(self,critic_gradients,states):
        self.sess.run(self.optimize,feed_dict={self.tf_critic_gradients:critic_gradients,self.tf_state_goal_input:states})

    def get_parameters(self):
        # tw1,tw2,tw3,tb1,tb2,tb3 = self.sess.run([self.w1,self.w2,self.w3,self.b1,self.b2,self.b3]) # TODO is sess.run() needed
        return [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]# (tw1,tw2,tw3,tb1,tb2,tb3)


############################################ Target Critic ############################################
class TargetCrtiticNet(CriticNet):
    def __init__(self,sess,learning_rate,tau):
        CriticNet.__init__(self,sess,learning_rate)
        self.tau = tau

    def set_params(self,new_params): # theres probably a better way!
        self.w1 = new_params[0]
        self.w2 = new_params[1]
        self.w2A = new_params[2]
        self.w3 = new_params[3]

    def update_params(self,new_params): # theres probably a better way!
        self.w1 = tf.multiply(new_params[0], self.tau) + tf.multiply((1-self.tau),self.w1)
        self.w2 = tf.multiply(new_params[1], self.tau) + tf.multiply((1-self.tau),self.w2)
        self.w2A = tf.multiply(new_params[2], self.tau) + tf.multiply((1-self.tau),self.w2A)
        self.w3 = tf.multiply(new_params[3], self.tau) + tf.multiply((1-self.tau),self.w3)
        self.b1 = tf.multiply(new_params[4], self.tau) + tf.multiply((1-self.tau),self.b1)
        self.b2 = tf.multiply(new_params[5], self.tau) + tf.multiply((1-self.tau),self.b2)
        self.b3 = tf.multiply(new_params[6], self.tau) + tf.multiply((1-self.tau),self.b3)


############################################ Target Actor ############################################

class TargetActorNet(object):
    def __init__(self,sess,learning_rate,action_bounds,tau):
        self.sess = sess
        self.learning_rate = learning_rate
        self.action_bounds = action_bounds
        self.tau = tau

        with tf.name_scope('Actor_Parameters'):
            self.w1 = tf.Variable(tf.random_uniform(shape=(NUM_INPUTS,NUM_HIDDEN1),minval = - 1/np.sqrt(NUM_INPUTS),maxval = 1/np.sqrt(NUM_INPUTS)))
            self.b1 = tf.Variable(tf.constant(0.01,shape =[NUM_HIDDEN1]))
            self.w2 = tf.Variable(tf.random_uniform(shape=(NUM_HIDDEN1,NUM_HIDDEN2),minval = - 1/np.sqrt(NUM_HIDDEN1),maxval = 1/np.sqrt(NUM_HIDDEN1)))
            self.b2 = tf.Variable(tf.constant(0.01,shape =[NUM_HIDDEN2]))
            self.w3 = tf.Variable(tf.random_uniform(shape=(NUM_HIDDEN2,NUM_ACTION),minval = - FINAL_ACTOR_RNGE,maxval = FINAL_ACTOR_RNGE))
            self.b3 = tf.Variable(tf.constant(0.01,shape = [NUM_ACTION]))


        with tf.name_scope('Actor_Inference'):
            self.tf_state_goal_input = tf.placeholder(tf.float32,shape=(None,NUM_INPUTS))

            z1 = tf.matmul(self.tf_state_goal_input,self.w1) + self.b1
            a1 = tf.nn.relu(z1)
            z2 = tf.matmul(a1,self.w2) + self.b2
            a2 = tf.nn.relu(z2)
            z3 = tf.matmul(a2,self.w3) + self.b3
            a3 = tf.nn.tanh(z3)
            self.y_ = tf.multiply(a3,self.action_bounds)

    def predict(self,states_goal_batch):
        y_ = self.sess.run([self.y_],feed_dict={self.tf_state_goal_input:states_goal_batch})
        return y_

    def set_params(self,new_params):
        self.w1 = new_params[0]
        self.w2 = new_params[2]
        self.w3 = new_params[4]

    def update_params(self,new_params):
        self.w1 = tf.multiply(new_params[0], self.tau) + tf.multiply((1-self.tau),self.w1)
        self.b1 = tf.multiply(new_params[1], self.tau) + tf.multiply((1-self.tau),self.b1)
        self.w2 = tf.multiply(new_params[2], self.tau) + tf.multiply((1-self.tau),self.w2)
        self.b2 = tf.multiply(new_params[3], self.tau) + tf.multiply((1-self.tau),self.b2)
        self.w3 = tf.multiply(new_params[4], self.tau) + tf.multiply((1-self.tau),self.w3)
        self.b3 = tf.multiply(new_params[5], self.tau) + tf.multiply((1-self.tau),self.b3)


############################################ Experience Memory ############################################
class Memory(object):
    def __init__(self,max_size):
        self.internal_mem = []
        self.max_size = max_size

    def sample_batch(self,batch_size):
        rnd_smpl_mem = random.sample(self.internal_mem,min(len(self.internal_mem),batch_size))
        temp_state0 = []
        temp_action = []
        temp_reward = []
        temp_state = []
        temp_done = []
        for mem in rnd_smpl_mem:
            temp_state0.append(mem[0])
            temp_action.append(mem[1])
            temp_reward.append(mem[2])
            temp_state.append(mem[3])
            temp_done.append(mem[4])
        return temp_state0, temp_action, temp_reward, temp_state, temp_done

        # return list(zip(*temp_mem))[0],list(zip(*temp_mem))[1],list(zip(*temp_mem))[2],list(zip(*temp_mem))[3],list(zip(*temp_mem))[4]

    def remember(self,new_data):
        self.internal_mem.append(new_data)
        if len(self.internal_mem) > self.max_size:
            self.internal.pop(0)
            self.internal_mem.pop(0)

############################################ Ornstein Uhlenbeck Action Noise ############################################
# from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class ActionNoise(object):
    def reset(self):
        pass
class OrnsteinUhlenbeckActionNoise(ActionNoise): # or replace 'ActionNoise' with 'object'
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)



############################################ INSTANCES ############################################
session = tf.Session()

# Epsiode Critic & Actor
critic = CriticNet(session,CRITIC_ALPHA)
actor = ActorNet(session,ACTOR_ALPHA,ACTION_BOUNDS)
# Target Critic & Actor
target_critic = TargetCrtiticNet(session,CRITIC_ALPHA,TAU)
target_actor  = TargetActorNet(session,ACTOR_ALPHA,ACTION_BOUNDS,TAU)

memory = Memory(MEMORY_SIZE)
ornuhl_noise = OrnsteinUhlenbeckActionNoise(OrnUhl_MEAN,OrnUhl_SIGMA)

session.run(tf.global_variables_initializer())
############################################ Environment Simulation ############################################

init_params_critic = critic.get_parameters()
init_params_actor = actor.get_parameters()

target_critic.set_params(init_params_critic)
target_actor.set_params(init_params_actor)

# env = gym.wrappers.FlattenDictWrapper(env, dict_keys=[′observation′, ′desired_goal′]) # Flatten the new 'Robotics' observation Dictonray into an array

for episode in range(NUM_EPISODES):
    state0 = env.reset()
    episode_rwd = 0
    t = 0
    done = False
    ornuhl_noise.reset()

    while not done:
        action = actor.predict([state0])[0][0]
        noise = ornuhl_noise.__call__().reshape(-1,4)[0]
        action += noise

        state, reward, done, _ = env.step(action)
        episode_rwd += reward
        t+= 1

        # Render
        if episode%RENDER_SKIP == 0:
            env.render()

        # save to memeory and get random minibatch
        memory.remember((state0,action,reward,state,done))
        state0_minibatch,action_minibatch,reward_minibatch,state_minibatch,done_minibatch = memory.sample_batch(MINIBATCH_SIZE)


        # get actor predictions for state+1, i.e next state
        trgt_actor_pred_minibatch = target_actor.predict(state_minibatch)[0]
        # get target critic predicted values for state and new action predictions
        trgt_q_state_minibatch = target_critic.predict(state_minibatch,trgt_actor_pred_minibatch)[0]
        # creat Y = r + Gamma*Q'(s,a) ;s,a are next step values
        Y_minibatch = reward_minibatch + GAMMA*trgt_q_state_minibatch


        # update critic Q
        critic.train(state0_minibatch,action_minibatch,Y_minibatch)

        # get actor predictions for state0
        trgt_actor_pred_minibatch = target_actor.predict(state0_minibatch)[0]
        # get critic gradients for actor Training
        critic_gradients = critic.get_gradients(state0_minibatch,trgt_actor_pred_minibatch)[0][0]

        # update actor  µ
        actor.train(critic_gradients,state0_minibatch)


        # get epsiode critic & actorparams
        new_params_critic = critic.get_parameters()
        new_params_actor = actor.get_parameters()
        # update target critic Q' & target actor  µ'
        target_critic.update_params(new_params_critic)
        target_actor.update_params(new_params_actor)


        state0 = state

        # just for LunarLander
        if t > 300:
            done == True

    print('Episode {} Reward {} Steps {}'.format(episode,episode_rwd,t))










#eop
