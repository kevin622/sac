Changed Things after clone coding
1. `replay_buffer.py` : changed memory to list
2. `models.py` : not using `ValueNetwork`, used xavier initialization, made `Q_network` output 2 values(Q1, Q2), applied clamping in policy std, `log_prob.sum` dimension change(-1 => 1)
3. `sac.py` : added alpha, used unsqueeze to reward and mask, used unsqueeze in input policy(for the `sum` dimesion change in `models.py`)
    By not using Value, the parameter update part has changed totally.
4. `main.py` : added alpha in args, `done` shouldn't be multiplied. Rather `not done` should be.(_This might have been a big problem!!!_)


# SAC(Soft Actor Critic) Implementation

This repository is for implementation of [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290).
## How to train

When specifying GPU, 
```bash
CUDA_VISIBLE_DEVICES=1,2 python train.py
CUDA_VISIBLE_DEVICES=1,2 python main.py --buffer_size 1000
CUDA_VISIBLE_DEVICES=1,2 python main_2.py
```


## About the paper

`Off-Policy` `Maximum Entropy` `Actor-Critic` algorithm

- Provides `sample efficient learning` and `stability`
- Extends readily to high-dimensional tasks(Humanoid Benchmark)

Objective: Maximum Entropy Objective
$$
J(\pi) = \sum^{T}_{t=0}E_{(s_t, a_t) \sim \rho_\pi} {\left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]}
$$

<img src="https://latex.codecogs.com/svg.image?\bg_white&space;J(\pi)&space;=&space;\sum^{T}_{t=0}E_{(s_t,&space;a_t)&space;\sim&space;\rho_\pi}&space;{\left[&space;r(s_t,&space;a_t)&space;&plus;&space;\alpha&space;\mathcal{H}(\pi(\cdot&space;|&space;s_t))&space;\right]}" title="\bg_white J(\pi) = \sum^{T}_{t=0}E_{(s_t, a_t) \sim \rho_\pi} {\left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right]}" />

### **Derivation of Soft Policy Iteration**

- Policy Evaluation Step

    Aim : Compute the value of a policy
    $$
    \text{repeat } Q \leftarrow \mathcal{T}^\pi Q \\
    \mathcal{T}^\pi Q(s_t, a_t) \triangleq r(s_t, a_t) + \gamma\mathbb{E}_{s_{t+1}\sim p}[V(s_{t+1})] \\ 
    V(s_t) = \mathbb{E}_{a_t \sim \pi}[Q(s_t, a_t) - \log{\pi(a_t|s_t)}] \\
    V(s_t) \text{ is the soft state value}
    $$
    <img src="https://latex.codecogs.com/svg.image?\text{repeat&space;}&space;Q&space;\leftarrow&space;\mathcal{T}^\pi&space;Q&space;\\\mathcal{T}^\pi&space;Q(s_t,&space;a_t)&space;\triangleq&space;r(s_t,&space;a_t)&space;&plus;&space;\gamma\mathbb{E}_{s_{t&plus;1}\sim&space;p}[V(s_{t&plus;1})]&space;\\&space;V(s_t)&space;=&space;\mathbb{E}_{a_t&space;\sim&space;\pi}[Q(s_t,&space;a_t)&space;-&space;\log{\pi(a_t|s_t)}]&space;\\V(s_t)&space;\text{&space;is&space;the&space;soft&space;state&space;value}" title="\text{repeat } Q \leftarrow \mathcal{T}^\pi Q \\\mathcal{T}^\pi Q(s_t, a_t) \triangleq r(s_t, a_t) + \gamma\mathbb{E}_{s_{t+1}\sim p}[V(s_{t+1})] \\ V(s_t) = \mathbb{E}_{a_t \sim \pi}[Q(s_t, a_t) - \log{\pi(a_t|s_t)}] \\V(s_t) \text{ is the soft state value}" />

- Policy Improvement Step
    Aim : Update the policy towards the exponential of the new Q-function

    For each state, update the policy according to below.
    $$
    \pi_{new} = \text{arg} \underset{\pi^\prime \in \Pi}{\text{ min }} D_{KL} \left(\pi^\prime(\cdot|s_t) \parallel \frac{exp(Q^{\pi_{old}}(s_t, \cdot))}{Z^{\pi_{old}}(s_t)} \right)
    $$
    
    <img src="https://latex.codecogs.com/svg.image?\pi_{new}&space;=&space;\text{arg}&space;\underset{\pi^\prime&space;\in&space;\Pi}{\text{&space;min&space;}}&space;D_{KL}&space;\left(\pi^\prime(\cdot|s_t)&space;\parallel&space;\frac{exp(Q^{\pi_{old}}(s_t,&space;\cdot))}{Z^{\pi_{old}}(s_t)}&space;\right)" title="\pi_{new} = \text{arg} \underset{\pi^\prime \in \Pi}{\text{ min }} D_{KL} \left(\pi^\prime(\cdot|s_t) \parallel \frac{exp(Q^{\pi_{old}}(s_t, \cdot))}{Z^{\pi_{old}}(s_t)} \right)" />

This algorithm  finds the optimal solution, it can be performed only in the tabular case.
Thus for continuous domains, we have to approximate the algorithm.

- A function approximator to represent Q values.
- Running two steps until convergence would be computationally to expensive.



### **Soft Actor Critic(SAC)**

Use function approximators for both `Q-function` and `policy`.
Alternate between optimizing both networks with stochastic gradient descent (instead of running evaluation and improvement to convergence) 
$$
\text{Function Approximators for } V_\psi(s_t), Q_\theta(s_t, a_t), \pi_\phi(a_t|s_t)
$$

<img src="https://latex.codecogs.com/svg.image?\text{Function&space;Approximators&space;for&space;}&space;V_\psi(s_t),&space;Q_\theta(s_t,&space;a_t),&space;\pi_\phi(a_t|s_t)" title="\text{Function Approximators for } V_\psi(s_t), Q_\theta(s_t, a_t), \pi_\phi(a_t|s_t)" />



### Gradients

- Soft Value

$$
\hat\triangledown_{\psi}J_V(\psi) = \triangledown_\psi V_\psi(s_t)(V_\psi(s_t) - Q_\theta (s_t, a_t) + \log{\pi_\phi (a_t|s_t)})
$$

<img src="https://latex.codecogs.com/svg.image?\hat\triangledown_{\psi}J_V(\psi)&space;=&space;\triangledown\psi&space;V_\psi(s_t)(V_\psi(s_t)&space;-&space;Q_\theta&space;(s_t,&space;a_t)&space;&plus;&space;\log{\pi_\phi&space;(a_t|s_t)})" title="\hat\triangledown_{\psi}J_V(\psi) = \triangledown\psi V_\psi(s_t)(V_\psi(s_t) - Q_\theta (s_t, a_t) + \log{\pi_\phi (a_t|s_t)})" />



- Soft Q function

$$
\hat\triangledown_{\theta}J_Q(\theta) = \triangledown_\theta Q_\theta(a_t, s_t) (Q_\theta(a_t, s_t) - r(s_t, a_t) - \gamma V_{\bar\psi}(s_{t+1})) \\
\bar\psi \text{ is an exponenetially moving average of the value network weights}
$$

<img src="https://latex.codecogs.com/svg.image?\hat\triangledown_{\theta}J_Q(\theta)&space;=&space;\triangledown_\theta&space;Q_\theta(a_t,&space;s_t)&space;(Q_\theta(a_t,&space;s_t)&space;-&space;r(s_t,&space;a_t)&space;-&space;\gamma&space;V_{\bar\psi}(s_{t&plus;1}))&space;\\\bar\psi&space;\text{&space;is&space;an&space;exponenetially&space;moving&space;average&space;of&space;the&space;value&space;network&space;weights}" title="\hat\triangledown_{\theta}J_Q(\theta) = \triangledown_\theta Q_\theta(a_t, s_t) (Q_\theta(a_t, s_t) - r(s_t, a_t) - \gamma V_{\bar\psi}(s_{t+1})) \\\bar\psi \text{ is an exponenetially moving average of the value network weights}" />



- Policy

$$
\hat\triangledown_{\phi}J_\pi(\phi) = \triangledown_\phi \log{\pi_\phi (a_t|s_t)} + \left( \triangledown_{a_t}\log{\pi_\phi (a_t|s_t)} - \triangledown_{a_t}Q(s_t, a_t) \right)\triangledown_\phi f_\phi(\epsilon_t;s_t) \\
a_t = f_\phi(\epsilon_t;s_t) \text{ : Policy followed by Neural Net Transformation}
$$

<img src="https://latex.codecogs.com/svg.image?\hat\triangledown_{\phi}J_\pi(\phi)&space;=&space;\triangledown_\phi&space;\log{\pi_\phi&space;(a_t|s_t)}&space;&plus;&space;\left(&space;\triangledown_{a_t}\log{\pi_\phi&space;(a_t|s_t)}&space;-&space;\triangledown_{a_t}Q(s_t,&space;a_t)&space;\right)\triangledown_\phi&space;f_\phi(\epsilon_t;s_t)&space;\\a_t&space;=&space;f_\phi(\epsilon_t;s_t)&space;\text{&space;:&space;Policy&space;followed&space;by&space;Neural&space;Net&space;Transformation}" title="\hat\triangledown_{\phi}J_\pi(\phi) = \triangledown_\phi \log{\pi_\phi (a_t|s_t)} + \left( \triangledown_{a_t}\log{\pi_\phi (a_t|s_t)} - \triangledown_{a_t}Q(s_t, a_t) \right)\triangledown_\phi f_\phi(\epsilon_t;s_t) \\a_t = f_\phi(\epsilon_t;s_t) \text{ : Policy followed by Neural Net Transformation}" />

### Hyperparameters

- Shared
    - optimizer: Adam
    - learning rate: 0.003
    - discount(gamma): 0.99
    - replay buffer size: 10^6
    - number of hidden layers: 2
    - number of hidden units per layer: 256
    - number of samples per minibatch: 256
    - Nonlinearity: ReLU
- SAC
    - target smoothing coefficient(tau): 0.005
    - target update interval: 1
    - gradient steps: 1
- SAC(hard target update)
    - target smoothing coefficient(tau): 1
    - target update interval: 1000
    - gradient seps(execpt humanoids): 4
    - gradient steps(humanoids): 1

### Environments

| Environment     | Action Dimensions | Reward Scale |
| --------------- | ----------------- | ------------ |
| Hopper-v1       | 3                 | 5            |
| Walker2d-v1     | 6                 | 5            |
| HalfCheetah-v1  | 6                 | 5            |
| Ant-v1          | 8                 | 5            |
| Humanoid-v1     | 17                | 20           |
| Humanoid(rllab) | 21                | 10           |

