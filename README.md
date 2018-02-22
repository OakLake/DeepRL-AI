# AI-Learning
Learning and Artificial Intelligence in Robotics
--------
### Contents:
- Fixed Topology Neural Network Search.
- Q-Learning.
- Deep RL Policy Network.
- Deep Q-Learning (DQN).
- Deep Q-Learning (DQN) with target network & frame skipping implementation.

--------
### Results:

##### MountainCar-v0:
- Using random weight search for fixed topology neural network.
    <p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/MountainCar/MountainCar_NN.gif">
    </p>
    
##### CartPole-v1:
- Tabular Q-Learning.
    <p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/CartPole_RL.gif">
    </p>
##### LunarLander-v2:
- Deep Q Learning with frame skipping(repeat same action for 3 frames) & target network updated at (epsiode%2==0).
    landing at epsiode 720:
    <p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/LunarLander/so_cool.gif">
    </p>
    
- Deep RL policy learning.
<p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/LunarLander/clever_girl.gif">
    </p>
    
<p align="center">
    <img src="https://github.com/OakLake/AI-Learning/blob/master/LunarLander/landing.gif">
    </p>
--------

### Interesting Resouces:

- Welcoming the Era of Deep Neuroevolution: https://eng.uber.com/deep-neuroevolution/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BzyxkMF5OTd%2BI48jAyJJ%2B2A%3D%3D
- MIT Deep-RL self-driving cars: https://selfdrivingcars.mit.edu
- Deep RL lecture by David Silver UCL: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Resources_files/deep_rl.pdf
- DQN Nature paper, 'Human-level control through deep reinforcement learning' (2015)
