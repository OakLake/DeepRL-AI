'''
''
'
    Deep Deterministic Policy Gradient :: https://arxiv.org/abs/1509.02971
'
''
TODO:
'''

import tensorflow as tf
import gym
import random
import numpy as np
import os
import tflearn


os.system('clear')


env_name = 'Reacher-v2'
env = gym.make(env_name)
print('############################################')
print('Action Space: ', env.action_space)
print('Action Space H : ', env.action_space.high)

print('Obsrev Space: ', env.observation_space)
print('Obsrev H: ',env.observation_space.high)
print('Obsrev L: ',env.observation_space.low)
print('############################################')


############################################ Algorithm Parameters ############################################

NUM_INPUTS    = 11
NUM_ACTION    = 2

GAMMA         = 0.99
CRITIC_ALPHA  = 1e-3
ACTOR_ALPHA   = 1e-4
ACTION_BOUNDS = np.array([env.action_space.high],dtype=np.float32)
NUM_EPISODES  = 1000
NUM_EPOCHS = 5000
MEMORY_SIZE   = 1e5
TAU           = 0.001
MINIBATCH_SIZE= 128
RENDER_SKIP   = 5
OrnUhl_SIGMA  = 0.05
OrnUhl_MEAN   = np.zeros((NUM_ACTION,1))



############################################ ActorNetwork & CriticNetwork ############################################

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.clipped_gradients = [tf.clip_by_norm(grad , 5.0) for grad in self.unnormalized_actor_gradients]
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.clipped_gradients))

        # Optimization Op
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32,shape=(None,self.s_dim))
        net = tf.layers.dense(inputs,400)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.layers.dense(net,300)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        out =  tf.layers.dense(net,self.a_dim,activation = tf.nn.tanh,kernel_initializer = w_init )
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32,shape=(None,self.s_dim))
        action = tf.placeholder(tf.float32,shape=(None,self.a_dim))
        net = tf.layers.dense(inputs,400)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tf.layers.dense(net,300)
        t2 = tf.layers.dense(action,300)

        net = tf.add(t1, t2)
        net = tf.layers.batch_normalization(net)
        # net = tf.matmul(net, t1.kernel) + tf.matmul(action, t2w.kernel) + t2.bias
        net = tf.nn.relu(net)


        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        out =  tf.layers.dense(net,1,kernel_initializer = w_init )
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


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

# Critic & Actor Networks
actor = ActorNetwork(session,NUM_INPUTS,NUM_ACTION,3,CRITIC_ALPHA,TAU,MINIBATCH_SIZE)
critic = CriticNetwork(session,NUM_INPUTS,NUM_ACTION,ACTOR_ALPHA,TAU,GAMMA,actor.get_num_trainable_vars())


memory = Memory(MEMORY_SIZE)
ornuhl_noise = OrnsteinUhlenbeckActionNoise(OrnUhl_MEAN,OrnUhl_SIGMA)

session.run(tf.global_variables_initializer())
############################################ Environment Simulation ############################################


avg100_score=[]

for epoch in range(NUM_EPOCHS):
    for episode in range(NUM_EPISODES):
        state0 = env.reset()
        episode_rwd = 0
        t = 0
        done = False
        ornuhl_noise.reset()

        while not done:

            # Render
            if episode%RENDER_SKIP == 0:
                env.render()

            if (epoch == 0) & (episode < 200) :
                action = env.action_space.sample()
            else:
                action = actor.predict([state0])[0][0]

                noise = ornuhl_noise.__call__().reshape(-1,NUM_ACTION)[0]
                action += noise

            state, reward, done, _ = env.step(action)
            episode_rwd += reward
            t+= 1



            # save to memeory and get random minibatch
            memory.remember((state0,action,reward,state,done))
            # if epoch > 5:
            state0_minibatch,action_minibatch,reward_minibatch,state_minibatch,done_minibatch = memory.sample_batch(MINIBATCH_SIZE)



            trgt_actor_pred_minibatch = actor.predict_target(state_minibatch)
            trgt_q_state_minibatch = critic.predict_target(state_minibatch,trgt_actor_pred_minibatch)

            # Y_ = np.array(reward_minibatch + GAMMA*trgt_q_state_minibatch).reshape(-1,1)
            # print(Y_.shape)
            # Y_minibatch = list(([x] for x in Y_))

            Y_minibatch = []

            for ix in range(len(state_minibatch)):
                if done_minibatch[ix]:
                    Y_minibatch.append(np.array(reward_minibatch[ix], dtype=np.float32))
                else:
                    Y_minibatch.append(reward_minibatch[ix] + GAMMA*trgt_q_state_minibatch[ix])

            Y_ = np.array(Y_minibatch).reshape(-1,1)

            critic.train(state0_minibatch,action_minibatch,Y_)
            actor_pred0_minibatch = actor.predict(state0_minibatch)

            critic_gradients = critic.action_gradients(state0_minibatch,actor_pred0_minibatch)[0]
            actor.train(state0_minibatch,critic_gradients)

            critic.update_target_network()
            actor.update_target_network()


            state0 = state

        avg100_score.append(episode_rwd)
        avg100_score = avg100_score[-100:]
        avg100 = np.mean(avg100_score)
        print('Epoch {} Episode {} Reward {} Steps {} <><> Avg100 Score {}'.format(epoch,episode,episode_rwd,t,avg100))










#eop
