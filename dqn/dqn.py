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
        i = 0
        for i in range(len(layers_sizes) - 1):
            setattr(self, f'layer_{i}', nn.Linear(layers_sizes[i], layers_sizes[i+1]))
            # self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i+1]))
        self.nb_layers = i + 1
        # self.myparameters = nn.ParameterList([layer.parameters() for layer in self.layers])

    def forward(self, x):
        for i in range(self.nb_layers):
        # for layer in self.layers:
            # x = F.relu(layer(x))
            x = F.relu(getattr(self, f'layer_{i}')(x))
        return x


class DQN(object):
    def __init__(self, state_size, action_size, model, learning_rate=1e-4, gamma=0.99, tau=1000):
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
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(list(self.model.parameters()), lr=self.learning_rate, momentum=0.9)

    def backward(self, states, actions, rewards, next_states, dones):
        q_values = self.model(states).gather(1, actions.long().view(-1, 1)).view(-1)
        # Bellman target
        next_q_values = torch.max(self.target_model(next_states), dim=-1)[0].detach()
        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.global_step += 1

        if self.tau < 1:
            # Update target (soft)
            target_state_dict = self.target_model.state_dict()
            for key in target_state_dict:
                target_state_dict[key] = (1 - self.tau) * self.model.state_dict()[key] \
                                         + self.tau * target_state_dict[key]
        elif self.global_step % self.tau == 0:
            # Update target (hard)
            self.target_model.load_state_dict(self.model.state_dict())

        return loss

    def forward(self, states):
        return torch.argmax(self.model(states))
