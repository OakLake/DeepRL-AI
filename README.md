# AI-Learning
Learning and Artificial Intelligence in Robotics
--------
### Contents:
- Fixed Topology Neural Network Search.
- Q-Learning.
- Deep RL Policy Network.
- Deep Q-Learning (DQN).
- Deep Q-Learning (DQN) +[target network, reward clipping, frame skipping].
- Deep Deterministic Policy Gradient (DDPG).

--------
### Results:

##### MountainCar-v0:
- Using random weight search for fixed topology neural network.
    <p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/GIFS/MountainCar_NN.gif">
    </p>
    
##### CartPole-v1:
- Tabular Q-Learning.
    <p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/GIFS/CartPole_RL.gif">
    </p>
##### LunarLander-v2:
- Deep Q Learning with frame skipping(repeat same action for 3 frames), target network updated at (epsiode%2==0) & reward clipping(-1,1).
    landing at epsiode 720:
    <p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/GIFS/so_cool.gif">
    </p>
    paremeters, for below:
    - refresh target net every 10 episodes.
    - skip 3 frames.
    - minibatch size 32.
    - at episode 460.
    
    <p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/GIFS/refreshNet10_skip3_batch32_frame460_NICE.gif">
    </p>
   
    
- Deep RL policy learning.
<p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/GIFS/clever_girl.gif">
    </p>
    
<p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/GIFS/landing.gif">
    </p>
    
##### InvertedPendulum-v2:
- Deep Deterministic Policy Gradient (DDPG).

InvertedPendulum_v2 & Pendulum_v0 are based on the same algorithm for different Gym envs, InvertedDoublePendulum_v2 uses an enhanced learning method.

This new change allows for faster learning, solving an issue where the algorithm would show suboptimal and non learning behaviour before suddnly increasing its score and learning, which might not occur at all!. The artifact of which is the algorithm sticking to a set actions, the new learning method performs much better by training the networks on a random policy [choosing random actions] before allowing the algoirthm to apply its action to the Gym env.

<p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/GIFS/InvertedPendulum_v2_DDPG_9x1000%2B720episodes_SOLVED.gif">
</p>
<p align="center">
    <img src="https://github.com/OakLake/DeepRL-AI/blob/master/GIFS/Pendulum_v0_8360.gif">
</p>


img links to youtube video:
[![alt text](https://github.com/OakLake/DeepRL-AI/blob/master/GIFS/Still_InvertedDoublePendulum_v2.jpg)](https://www.youtube.com/watch?v=fXbqDDaJDvg&feature=youtu.be "InvertedDoublePendulum")

Link: https://youtu.be/fXbqDDaJDvg

--------

### Interesting Resouces:

- Welcoming the Era of Deep Neuroevolution: https://eng.uber.com/deep-neuroevolution/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BzyxkMF5OTd%2BI48jAyJJ%2B2A%3D%3D
- MIT Deep-RL self-driving cars: https://selfdrivingcars.mit.edu
- Deep RL lecture by David Silver UCL: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf
- DQN Nature paper, 'Human-level control through deep reinforcement learning' (2015)
- DDPG Paper, 'Continuous control with deep reinforcement learning' https://arxiv.org/abs/1509.02971
