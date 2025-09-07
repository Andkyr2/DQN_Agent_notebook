# Deep Q-Network (DQN) for CartPole Balancing

This project implements a Deep Q-Network (DQN) agent to solve the classic reinforcement learning environment, CartPole-v1, using PyTorch and Gymnasium. The goal is to train an agent to balance a pole on top of a moving cart by applying forces to the left or right.

## Overview of the CartPole Environment

* **Objective:** Prevent the pole from falling over and keep the cart from moving too far off-center.
* **State Space ($s$):** A continuous state represented by four values:
    1.  Cart Position
    2.  Cart Velocity
    3.  Pole Angle
    4.  Pole Angular Velocity
* **Action Space ($a$):** Two discrete actions:
    * 0: Push cart to the left.
    * 1: Push cart to the right.
* **Reward ($r$):** A reward of +1 is given for every timestep the pole remains upright. The episode ends if the pole angle exceeds a certain threshold, the cart moves too far from the center, or the maximum number of steps (500 in this version) is reached.

## Core Concepts: From Q-Learning to DQN

### 1. Q-Learning Basics

The foundation of DQN is **Q-Learning**, an algorithm designed to find the optimal policy by learning an **action-value function**, denoted as $Q^*(s, a)$. This function estimates the total expected future reward an agent will receive by taking action $a$ from state $s$ and then following the optimal policy thereafter.

In traditional Q-Learning with discrete states, this function is updated using the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)$$

Where:
* $\alpha$ (alpha) is the learning rate.
* $\gamma$ (gamma) is the discount factor, determining the importance of future rewards.
* $r + \gamma \max_{a'} Q(s', a')$ is the **TD Target**, representing a better estimate of the value of $Q(s, a)$.

### 2. Deep Q-Network (DQN)

The CartPole environment has a continuous state space, making it impractical to use a simple table to store Q-values for every possible state. DQN solves this by using a neural network as a function approximator to estimate the Q-value function: $Q(s, a; \theta) \approx Q^*(s, a)$, where $\theta$ represents the network's weights.

The network takes the state as input and outputs a vector of Q-values, one for each possible action.

### 3. Key Techniques for Stability

Training a neural network with reinforcement learning data can be unstable. DQN introduces two key techniques to mitigate this:

#### a. Experience Replay

Instead of training the network on consecutive experiences as they occur, we store transitions `(state, action, reward, next_state, done)` in a cyclical memory buffer. During training, we sample random mini-batches from this buffer.

**Why?** This breaks temporal correlations between consecutive samples, making the training data more Independent and Identically Distributed (IID), which significantly improves training stability for deep learning models.

#### b. Target Network

DQN uses two separate neural networks:
1.  **Policy Network ($Q_{policy}$):** The main network that decides which action to take and is actively updated during training.
2.  **Target Network ($Q_{target}$):** A clone of the policy network whose weights are held constant for a period.

The loss function aims to minimize the difference between the prediction from the policy network and the target value calculated using the target network.

* **Current Q-value:** $Q_{current} = Q_{policy}(s, a)$
* **Target Q-value:** $y = r + \gamma \max_{a'} Q_{target}(s', a')$

**Why?** If we used only one network, the target value $y$ would change at every training step because the network weights change. This creates a "moving target problem." By freezing the target network, we provide a stable target for several steps, allowing the policy network to converge more reliably before the target is updated. The target network's weights are periodically updated by copying the weights from the policy network.

### 4. Exploration vs. Exploitation ($\epsilon$-greedy)

To ensure the agent explores the environment sufficiently instead of getting stuck in a suboptimal policy, we use an **epsilon-greedy strategy**.
* With probability $\epsilon$, choose a random action (exploration).
* With probability $1-\epsilon$, choose the action with the highest Q-value according to the network (exploitation).

The value of $\epsilon$ starts high (e.g., 1.0) and gradually decays towards a small minimum value as training progresses, shifting focus from exploration to exploitation.

## Code Structure Breakdown

### 1. Environment Setup and Imports
* Installs necessary libraries (`gymnasium`, `torch`).
* Initializes the `CartPole-v1` environment.

### 2. Neural Network Definition (`DQN` class)
* A simple fully connected neural network defined using `torch.nn.Module`.
* **Architecture:** Input layer (state size = 4) -> Hidden layers (e.g., 512, 128, 64 neurons) -> Output layer (action size = 2).
* The forward pass takes a state tensor and returns the Q-values for "left" and "right" actions.

### 3. Replay Memory (`replay_memory` class)
* Implemented using `collections.deque` with a maximum capacity.
* `append(item)`: Adds a new experience transition to the memory.
* `sample(batch_size)`: Returns a random sample of transitions for batch training.

### 4. DQN Agent (`DQNAgent` class)

This class ties all components together:

* **Initialization (`__init__`)**:
    * Initializes hyperparameters: learning rate (`lr`), discount factor (`gamma`), initial exploration rate (`epsilon`), epsilon decay rate, and minimum epsilon.
    * Instantiates two `DQN` networks: `self.policy_net` and `self.target_net`.
    * Initializes the replay memory and the Adam optimizer.
* **Action Selection (`select_action`)**: Implements the $\epsilon$-greedy policy.
* **Target Network Update (`update_target_network`)**: Copies weights from `policy_net` to `target_net` using `load_state_dict()`.
* **Optimization (`optimize_model`)**: The core training logic.
    1.  Samples a mini-batch from `replay_memory`.
    2.  Calculates the **target Q-values ($y$)** for the batch using the `target_net`. The next state's value is zeroed out if the episode ended (`done = True`).
    3.  Calculates the **current Q-values ($Q_{current}$)** predicted by the `policy_net`.
    4.  Computes the Mean Squared Error (MSE) loss between $y$ and $Q_{current}$.
    5.  Performs backpropagation and updates the weights of the `policy_net`.
    6.  Decays `epsilon`.

### 5. Training and Evaluation Loop

* **Training Loop:** Iterates for a set number of episodes.
    * In each step of an episode, the agent selects an action using `select_action`.
    * The action is performed in the environment to receive `next_state`, `reward`, and `done` status.
    * The transition `(state, action, reward, next_state, done)` is stored in the replay memory.
    * `optimize_model()` is called to train the network.
    * The target network is updated every `target_update` episodes.
* **Evaluation:** After training, the agent is run again with exploration turned off ($\epsilon=0$) to evaluate its learned optimal policy. The episode rewards are plotted to visualize performance improvement over time.
