import numpy as np
import gymnasium as gym
import os
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from matplotlib import animation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Initialize the environment
env = gym.make('LunarLander-v3') #render_mode="human"

state_space = env.observation_space.shape[0]
print('State Space:', state_space)
action_space = env.action_space.n
print('Action Space:', action_space)

import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# Policy Network

class Policy(nn.Module):
    def __init__(self , s_size , a_size , h_size ):
        super (Policy , self ).__init__ ()
        self.fc1 = nn.Linear( s_size , h_size )
        self.fc2 = nn.Linear( h_size , h_size * 2)
        self.fc3 = nn.Linear( h_size * 2, a_size )
    def forward(self , x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim =1)
    def act(self, state ):
        state = torch.from_numpy(state).float().unsqueeze(0)  #.to(device)
        probs = self.forward(state) # .cpu()
        m = Categorical(probs)
        # Random action
        action = m.sample()
        return action.item() , m.log_prob(action)
    
class Policy_QNN(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy_QNN, self).__init__()
        # self.fc1 = nn.Linear(s_size, h_size)
        self.activate_func = nn.ReLU()

        self.n_wires = 4
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.encoder_gates_rx = [tqf.rx] * self.n_wires
        self.encoder_gates_ry = [tqf.ry] * self.n_wires
        self.encoder_gates_rz = [tqf.rz] * self.n_wires
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        # self.crx0 = tq.CRX(has_params=True, trainable=True)


    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        actor_input = torch.from_numpy(actor_input).float().unsqueeze(0) 
        if actor_input.ndim == 4:
            batch_size, episode_limit, N, actor_input_dim = actor_input.shape
            x = actor_input.reshape(batch_size * episode_limit * N, actor_input_dim)
        else:
            x = actor_input
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        for k, gate in enumerate(self.encoder_gates_rx):
            gate(qdev, wires=k, params=x[:, k])
        for k, gate in enumerate(self.encoder_gates_ry):
            gate(qdev, wires=k, params=x[:, k+4])
        # for k, gate in enumerate(self.encoder_gates_rz):
        #     gate(qdev, wires=k, params=x[:, k])
        
        # Apply RX on all 10 qubits
        for i in range(qdev.n_wires):
            self.rx0(qdev, wires=i)

        # # Apply RZ on all 10 qubits
        for i in range(qdev.n_wires):
            self.rz0(qdev, wires=i)
        
        #Apply CNOT between every two adjacent qubits
        for i in range(0, qdev.n_wires-1, 1):
            qdev.cnot(wires=[i, i+1])

        # Apply RX on all 10 qubits
        # for i in range(qdev.n_wires):
        #     self.rx0(qdev, wires=i)

        # # Apply RX on all 10 qubits
        # for i in range(qdev.n_wires):
        #     self.ry0(qdev, wires=i)

        # # Apply RZ on all 10 qubits
        for i in range(qdev.n_wires):
            self.rz0(qdev, wires=i)

        state = qdev.states
        probs = self.measure(qdev)
        probs = F.softmax(probs, dim=-1)
        
        batch = state.shape[0]
        if actor_input.ndim == 4:
            probs = probs.reshape(batch_size, episode_limit, N, self.n_wires)

        # reshape to [batch, 2**n_wires]
        # state_flat = state.reshape(batch, 2 ** self.n_wires)
        # probs = ((state_flat.abs() ** 2).real)
        # if actor_input.ndim == 4:
        #     probs = probs.reshape(batch_size, episode_limit, N, 2 ** self.n_wires)

        m = Categorical(probs)
        action = m.sample()
        return action.item() , m.log_prob(action)
    
        # return probs    
    
# Training Function
def reinforce(
        policy ,
        optimizer ,
        n_training_episodes ,
        max_steps ,
        gamma ,
        print_every
        ):
    # scores_deque = deque(maxlen =100)
    scores = []

    # Each Episode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()[0]

        # t=1, 2, … , T (compute log(policy(a_t|s_t)))
        for t in range(max_steps):
            action , log_prob = policy(state)
            saved_log_probs.append(log_prob)
            state , reward , done , _ , _ = env.step(action)
            rewards.append(reward)
            if done :
                break
        # scores_deque.append(sum( rewards ))
        scores.append(sum(rewards))

        returns = deque(maxlen = max_steps)
        n_steps = len(rewards)

        # List of discounted Returns (compute gamma^t*G_t)
        for t in range(n_steps)[:: -1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma*disc_return_t + rewards[t])

        # Total loss (disc_return = gamma^t*G_t; log_prob = log(policy(a_t|s_t)))
        policy_loss = []
        for log_prob , disc_return in zip( saved_log_probs , returns ):
            policy_loss.append(-log_prob * disc_return )
        policy_loss = torch.cat( policy_loss ).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(" Episode {}, Reward : {}".format( i_episode ,sum(rewards)))

    return scores

# Hyperparameter
h_size = 128
lr = 0.001
n_training_episodes = 2000
max_steps = 1000
gamma = 0.99

policy = Policy_QNN(
        s_size = state_space ,
        a_size = action_space ,
        h_size = h_size ,
        ) #.to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

scores = reinforce (
        policy ,
        optimizer ,
        n_training_episodes ,
        max_steps ,
        gamma ,
        print_every = 100)

# Plotting the rewards per episode
import matplotlib.pyplot as plt
plt.plot(scores)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('REINFORCE on LunarLander')
plt.show()