# Introduction

This is my report of project 1: Navigation. In this project, I trained an agent to navigate in a large, square world.  The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The episodic task is considered solved when the agent gets an average score of +13 over 100 consecutive episodes.



# Algorithm

I solved the problem using Deep Q-Learning algorithm ([DQN]( https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf )) .

Pseudocode of DQN:

![DQN](F:\DRLND\deep-reinforcement-learning\p1_navigation\pic\DQN.png)

Hyperparameters:

​		buffer size: 100000

​		batch size: 64

​		learning rate: 0.0005

​		$\gamma$ (discounting rate): 0.99

​		$\tau$ (soft update coefficient): 0.001

​		number of time steps that model learns: 4

​		$\epsilon$-greedy parameters: 1.0 (start), 0.01 (end), 0.999 (decay)

Architecture of the neural network:

​		input layer (# 37) -> hidden layer 1 (# 128) -> hidden layer 2 (# 64) -> output layer (# 4)

# Result

![plot](F:\DRLND\deep-reinforcement-learning\p1_navigation\pic\DQN_curve.png)

The task was solved after 2354 episodes with average score of 13.04.

# Future work

Implement Double DQN, Dueling DQN, and Prioritized Experience Replay.