import gym
import numpy as np
import os
import math,random
import Box2D
import tensorflow as tf

os.system('clear')
print('LunarLander Deep Learning')
envName = 'LunarLander-v2'
env = gym.make(envName)

render_FLAG = True

'''
TF ::
'''

NUM_INPUTS  = 8
NUM_HIDDEN1 = 40
NUM_HIDDEN2 = 40
NUM_OUTPUT  = 4


# learning_rate= 5e-4
gamma=0.99
epsilon = 1
learning_rate = 0.001

graph = tf.Graph()

for batch_size in [32,64]:
    for refreshNet in [2,20]:

        with graph.as_default():

            # Inputs:
            with tf.name_scope('Inputs'):
                tf_X = tf.placeholder(tf.float32,shape=(None,NUM_INPUTS),name='tf_X') # 3: (state,action,reward)
                tf_Y = tf.placeholder(tf.float32,shape=(None,NUM_OUTPUT),name='tf_Y') # 4:(left,right,bottom,none) 'boosters'

                dropout = tf.placeholder_with_default(1.0, shape=(),name='dropoutVariable')
                cost = tf.placeholder(tf.float32,shape=(),name='cost')

            # Varaibles

            with tf.name_scope('Variables'):

                W1 = tf.Variable(tf.truncated_normal([NUM_INPUTS,NUM_HIDDEN1],stddev=0.35),name='W1')
                B1 = tf.Variable(tf.constant(0.,shape=[NUM_HIDDEN1]),name='B1')

                W2 = tf.Variable(tf.truncated_normal([NUM_HIDDEN1,NUM_HIDDEN2],stddev=0.35),name='W2') #,stddev=0.4
                B2 = tf.Variable(tf.constant(0.,shape=[NUM_HIDDEN2]),name='B2')

                W3 = tf.Variable(tf.truncated_normal([NUM_HIDDEN2,NUM_OUTPUT],stddev=0.35),name='W3')
                B3 = tf.Variable(tf.constant(0.,shape=[NUM_OUTPUT]),name='B3')

            # Dimension check

            print('tf_X: ',tf_X.shape)
            print('tf_Y: ',tf_Y.shape)

            print('W1: ',W1.shape)
            print('B1: ',B1.shape)
            print('W2: ',W2.shape)
            print('B2: ',B2.shape)
            print('W3: ',W3.shape)
            print('B3: ',B3.shape)

            print('NN Architecture: {}x{}x{}x{}'.format(NUM_INPUTS,NUM_HIDDEN1,NUM_HIDDEN2,NUM_OUTPUT))

            with tf.name_scope('Inputs_TargetN'):

                W1_t = tf.placeholder(tf.float32,shape=(NUM_INPUTS,NUM_HIDDEN1),name='W1_t')
                B1_t = tf.placeholder(tf.float32,shape=(NUM_HIDDEN1),name='B1_t')
                W2_t = tf.placeholder(tf.float32,shape=(NUM_HIDDEN1,NUM_HIDDEN2),name='W2_t')
                B2_t = tf.placeholder(tf.float32,shape=(NUM_HIDDEN2),name='B2_t')
                W3_t = tf.placeholder(tf.float32,shape=(NUM_HIDDEN2,NUM_OUTPUT),name='W3_t')
                B3_t = tf.placeholder(tf.float32,shape=(NUM_OUTPUT),name='B3_t')

            # Feed Forward
            with tf.name_scope('FeedForward_N'):
                A1_t = tf.matmul(tf_X,W1_t) + B1_t
                R_t = tf.nn.relu(A1_t)
                A2_t = tf.matmul(R_t,W2_t) + B2_t
                R_t = tf.nn.relu(A2_t)
                logits_target = tf.matmul(R_t,W3_t) + B3_t


            # Feed Forward
            with tf.name_scope('FeedForward'):
                A1 = tf.matmul(tf_X,tf.nn.dropout(W1,dropout)) + B1
                R = tf.nn.relu(A1)
                A2 = tf.matmul(R,tf.nn.dropout(W2,dropout)) + B2
                R = tf.nn.relu(A2)
                logits = tf.matmul(R,W3) + B3

            with tf.name_scope('Training'):

                predictions = tf.nn.softmax(logits)
                loss = tf.reduce_mean(tf.square(tf_Y-logits)) #tf.losses.huber_loss(labels=tf_Y,predictions=logits)    #
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

            tf.summary.scalar('Score',cost)
            tf.summary.scalar('Loss',loss)
            tf.summary.histogram('W1',W1)
            tf.summary.histogram('W2',W2)
            tf.summary.histogram('Activations',W3)
            merged = tf.summary.merge_all()



        with tf.Session(graph=graph) as session:

            tf.global_variables_initializer().run()
            dirName = './summary/'+envName+'_Alpha'+str(learning_rate)+'_batchSize'+str(batch_size)+'_refreshNet'+str(refreshNet)
            writer = tf.summary.FileWriter(dirName,session.graph)
            saver = tf.train.Saver()

            scores = []
            memory = []

            D = 100000
            max_trial = 1000
             # ~0.01% of memory to train on each time step >> for 1 full episode of 1000 trials whole memory retrain

            for episode in range(5000):
                print('========')
                no_op_max = 10
                no_op = 0 #np.random.randint(10)
                epsilon = np.max(((-1/400 *episode + 1),0.02)) #*= eDecay
                state0 = env.reset()
                rtn = 0
                t = -1
                if (episode%20 ==0) and (render_FLAG):
                    print('\033[92m' + '# RENDERING #' + '\033[0m')

                while(True):
                    t += 1

                    # rendering or not
                    if (episode%20 == 0) and (render_FLAG):
                        env.render()

                    # take action
                    if (np.random.random() < epsilon):
                        action = env.action_space.sample()
                    elif t <= no_op:
                        action = 0 # do nothing action
                    else:
                        action_V = session.run([predictions],feed_dict = {tf_X:[state0],dropout:1.})[0][0]
                        action = np.argmax(action_V)
                    for skip in range(2):
                        state, reward, done, _ = env.step(action)
                        rtn += reward
                        reward = np.clip(reward,-1,1) # Clipping rewards

                        # save to memory:
                        memory.append((state0,action,reward,state,done))

                        # update states
                        state0 = state

                    # Learning
                    # updating target network:
                    if (episode%refreshNet == 0):# or (episode<200):
                        # save old network weights
                        w1,b1,w2,b2,w3,b3 = session.run([W1,B1,W2,B2,W3,B3])

                    learn_memory = random.sample(memory,min(len(memory),batch_size))
                    batch_inputs = []
                    batch_outputs = []

                    for learn_memory in learn_memory:
                        old_state = learn_memory[0]
                        action = learn_memory[1]
                        reward = learn_memory[2]
                        state = learn_memory[3]
                        done_chck = learn_memory[4]

                        feed_dict = {tf_X:[old_state],W1_t:w1,B1_t:b1,W2_t:w2,B2_t:b2,W3_t:w3,B3_t:b3}
                        outputs = session.run([logits_target],feed_dict=feed_dict)[0][0]
                        observed_reward = reward
                        if done_chck is False:
                            feed_dict = {tf_X:[state],W1_t:w1,B1_t:b1,W2_t:w2,B2_t:b2,W3_t:w3,B3_t:b3}
                            future_rewards = session.run([logits_target],feed_dict = feed_dict)[0][0]
                            expected_reward = np.amax(future_rewards)
                            observed_reward = reward + gamma * expected_reward

                        outputs[action] = observed_reward

                        batch_inputs.append(old_state)
                        batch_outputs.append(outputs)


                    feed_dict = {tf_X:batch_inputs,tf_Y:batch_outputs,dropout:1.,cost:rtn}
                    _ ,l,summary = session.run([optimizer,loss,merged],feed_dict=feed_dict)




                    memory = memory[-D:] # keep memory at size D:10,000
                    # finished Play if done
                    if done or (t >= max_trial):
                            print('##....Episode: ',episode)
                            print('##....Reward : ',rtn)
                            print('##....Steps  : ',t)
                            print('##....Avg100 : ',np.mean(scores[-100:]))
                            print('##....Epsilon: ',epsilon)
                            scores.append(rtn)
                            break

                writer.add_summary(summary,episode)

            save_path = saver.save(session, "./sess/model.ckpt")

print('DONE !!!')
