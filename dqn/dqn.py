import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from copy import copy


class MLP(nn.Module):
    def __init__(self, layers_sizes):
        super().__init__()

        self.layers = []

        for i in range(len(layers_sizes) - 1):
            self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))


class DQN(object):
    def __init__(self, state_size, action_size, model, learning_rate=1e-4, gamma=0.99, tau=2500):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        assert(0 <= gamma <= 1)
        self.tau = tau
        assert(0 < tau)
        if self.tau >= 1:
            self.tau = int(self.tau)
        self.model = model

        self.global_step = 0

        self.target_model = copy(self.model)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def backward(self, states, actions, next_states, rewards):
        self.optimizer.zero_grad()

        q_values = self.model(states)[:, actions]
        # Bellman target
        next_q_values = np.max(self.target_model(next_states), axis=-1)
        target = rewards + self.gamma * next_q_values

        loss = self.criterion(q_values, target)
        loss.backward()
        self.optimizer.step()

        self.global_step += 1

        if self.tau < 1:
            # Update target (soft)
            pass
        elif self.global_step % self.tau == 0:
            # Update target (hard)
            pass

    def forward(self, states):
        return torch.argmax(self.model(states))
