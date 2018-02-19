import gym
import numpy as np
import pickle,os
import matplotlib.pyplot as plt
import math,random
import Box2D
import tensorflow as tf

os.system('clear')
print('LunarLander Deep Learning')

env = gym.make('LunarLander-v2')

# env = gym.wrappers.Monitor(env, directory, video_callable=lambda episode_id: True)


# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

def one_hot(x,ab):
    return np.array(np.arange(ab) == x,dtype=np.int32)

def reform_rewards(R_list):
    batch_RDecayed = list(R_list)
    cumulative = 0
    for r in reversed(range(len(batch_R))):
        cumulative = cumulative * gamma + batch_R[r]
        batch_RDecayed[r] = cumulative
    # normalize
    batch_RDecayed -= np.mean(batch_RDecayed)
    batch_RDecayed /= np.std(batch_RDecayed)
    return batch_RDecayed
'''
TF ::
'''

NUM_INPUTS  = 8
NUM_HIDDEN1 = 10
NUM_HIDDEN2 = 8
NUM_HIDDEN3 = 10
NUM_HIDDEN4 = 5
NUM_OUTPUT  = 4


# learning_rate= 5e-4
gamma=0.98
epsilon = 0.01

for learning_rate in [0.01]:
    graph = tf.Graph()

    with graph.as_default():
            # Inputs:
            with tf.name_scope('Inputs'):
                tf_X = tf.placeholder(tf.float32,shape=(None,NUM_INPUTS),name='tf_X') # 3: (state,action,reward)
                tf_Y = tf.placeholder(tf.float32,shape=(None,NUM_OUTPUT),name='tf_Y') # 4:(left,right,bottom,none) 'boosters'
                tf_DRN = tf.placeholder(tf.float32,shape=(None),name='tf_DRN') # depends on length of episode
                dropout = tf.placeholder_with_default(1.0, shape=(),name='dropoutVariable')
                cost = tf.placeholder(tf.float32,shape=(),name='cost')

            # Varaibles

            with tf.name_scope('Variables'):

                W1 = tf.Variable(tf.random_normal([NUM_INPUTS,NUM_HIDDEN1],stddev=0.4),name='W1')
                B1 = tf.Variable(tf.constant(0.,shape=[NUM_HIDDEN1]),name='B1')

                W2 = tf.Variable(tf.random_normal([NUM_HIDDEN1,NUM_HIDDEN2],stddev=0.4),name='W2')
                B2 = tf.Variable(tf.constant(0.,shape=[NUM_HIDDEN2]),name='B2')

                W3 = tf.Variable(tf.random_normal([NUM_HIDDEN2,NUM_OUTPUT],stddev=0.4),name='W3')
                B3 = tf.Variable(tf.constant(0.,shape=[NUM_OUTPUT]),name='B3')


            # Dimension check

            print('tf_X: ',tf_X.shape)
            print('tf_Y: ',tf_Y.shape)
            print('tf_DRN: ',tf_DRN.shape)

            print('W1: ',W1.shape)
            print('B1: ',B1.shape)
            print('W2: ',W2.shape)
            print('B2: ',B2.shape)
            print('W3: ',W3.shape)
            print('B3: ',B3.shape)


            # Feed Forward
            with tf.name_scope('FeedForward'):
                A1 = tf.matmul(tf_X,tf.nn.dropout(W1,dropout)) + B1
                Relu = tf.nn.relu(A1)
                A2 = tf.matmul(Relu,tf.nn.dropout(W2,dropout)) + B2
                Relu = tf.nn.relu(A2)
                A3 = tf.matmul(Relu,W3) #+ B3


            a_prob = tf.nn.softmax(A3) #[0.1 0.1 0.1 0.7]

            # [0.1 0.1 0.1 0.7](*)[0 0 0 1] -> expected Q(S,a) from NN, i.e predicted Q value
            # expected Q value for the state0 and action(encoded with one_hot) is given by tf_DRN

            with tf.name_scope('Training'):

                # x_ent = tf.nn.softmax_cross_entropy_with_logits(labels = tf_Y,logits=A3)

                # loss = tf.reduce_mean(softmax_x_entropy*tf_DRN)# + 0.001*tf.nn.l2_loss(W1) + 0.001*tf.nn.l2_loss(W2) + 0.001*tf.nn.l2_loss(W3)


                diff = tf.subtract(A3,tf.reshape(tf_DRN,[-1,1]))
                v_red = tf.multiply(diff,tf_Y)
                no_zeros = tf.reduce_sum(v_red,1) #https://www.tensorflow.org/versions/r0.12/api_docs/python/math_ops/reduction

                loss = tf.reduce_mean(tf.square(no_zeros))

                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            tf.summary.scalar('Score',cost)
            tf.summary.scalar('Loss',loss)
            tf.summary.histogram('W1',W1)
            tf.summary.histogram('W2',W2)
            tf.summary.histogram('Activations',A3)

            merged = tf.summary.merge_all()

    with tf.Session(graph=graph) as session:



        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter('./summary/Alpha'+str(learning_rate),session.graph)
        scores = []


        batch_X = []
        batch_Y = []
        batch_R = []
        batch_NS = []

        for episode in range(8000):
            print('========')



            observation = env.reset()
            state0, reward, done, _ = env.step(env.action_space.sample())
            rtn = 0
            t = -1
            while(True):
                t += 1
            # for t in range(500):
                if episode%50 == 0:
                    env.render()

                if (np.random.random() < epsilon):
                    action = env.action_space.sample()
                else:
                    action_prob = session.run([a_prob],feed_dict = {tf_X:[state0],dropout:1.})[0][0]
                    action = np.argmax(action_prob)
                    # action = np.random.choice(range(NUM_OUTPUT), p = action_prob) # biased sampling of actions, returns 0,1,2,3

                state, reward, done, _ = env.step(action)
                rtn += reward

                # save frame to memory [s,a,r,s']
                batch_X.append(state0)
                batch_Y.append(one_hot(action,NUM_OUTPUT))
                batch_R.append(reward)
                batch_NS.append(state)


                state0 = state

                if done or (t >= 500):
                        print('Episode: ',episode+1)
                        print('Steps  : ',t)
                        print('Reward : ',rtn)
                        print('Avg100 : ',np.mean(scores[-100:]))
                        scores.append(rtn)
                        break



            # Training Network for Value Function

            batch_size = np.min([1024,len(batch_X)])

            temp_x,temp_y,temp_r = [],[],[]
            for b in range(batch_size):
                # sample experience dataset
                i = np.random.randint(len(batch_X)) # is len(batchX) correcty?
                temp_x.append(batch_X[i])
                temp_y.append(batch_Y[i])

                maxQ = -100
                # for a in [0,1,2,3]:
                Qnext = session.run([A3],feed_dict = {tf_X:[batch_NS[i]],dropout:1.})[0][0] # gets Q(S',a') in one pass
                maxQ = np.max([maxQ,np.max(Qnext)])

                temp_r.append(batch_R[i] + gamma*maxQ)


            feed_dict = {tf_X:temp_x,tf_Y:temp_y,tf_DRN:temp_r,dropout:1.,cost:rtn}
            _ ,l,summary = session.run([optimizer,loss,merged],feed_dict=feed_dict)

            # print('Training Network')
            # print('loss: ',l)
            # print('Trained..')








            # Initialize episode rewards
            # shuffle experience
            # combined = list(zip(batch_X,batch_Y ,batch_RD))
            # random.shuffle(combined)
            #
            # batch_X[:], batch_Y[:], batch_RD[:] = zip(*combined)

            # Train network after episode




            writer.add_summary(summary,episode)




print('DONE !!!')
