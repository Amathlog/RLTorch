import itertools
from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim


class DDPG(object):
    def __init__(self, state_size, action_size, critic_model, actor_model, action_support = None,
                 learning_rate=(1e-5, 1e-4), gamma=0.99, tau=2500):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        assert(0 <= gamma <= 1)
        self.tau = tau
        assert(0 < tau)
        if self.tau >= 1:
            self.tau = int(self.tau)
        self.critic_model = critic_model
        self.actor_model = actor_model

        if action_support is None:
            self.action_scale = lambda x: x
        else:
            min_actions = action_support[0]
            max_actions = action_support[1]
            self.action_scale = lambda x: (max_actions - min_actions) * 0.5 * x + (max_actions + min_actions) * 0.5

        self.global_step = 0

        self.target_critic_model = copy(self.critic_model)
        self.target_critic_model.load_state_dict(self.critic_model.state_dict())
        self.target_critic_model.eval()

        self.target_actor_model = copy(self.actor_model)
        self.target_actor_model.load_state_dict(self.actor_model.state_dict())
        self.target_actor_model.eval()

        self.criterion = nn.MSELoss()
        self.critic_optimizer = optim.RMSprop(list(self.critic_model.parameters()), lr=self.learning_rate[0])
        self.actor_optimizer = optim.RMSprop(list(self.actor_model.parameters()), lr=self.learning_rate[1])

    def backward(self, states, actions, rewards, next_states, dones):
        # Convert numpy arrays to Tensors
        with torch.no_grad():
            states, actions, rewards, next_states, dones = \
                map(torch.from_numpy, [states, actions, rewards, next_states, dones])

        q_values = self.critic_model(torch.cat((states, actions), -1)).squeeze(dim=-1)
        # Bellman target
        next_q_values = self.target_critic_model(torch.cat((next_states, self.target_actor_model(next_states)), -1)).squeeze(dim=-1).detach()
        target = rewards + self.gamma * next_q_values * (1 - dones)

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        critic_loss = self.criterion(q_values, target)
        actor_loss = -self.critic_model(torch.cat((states, self.actor_model(states)), -1))
        actor_loss = actor_loss.mean()

        critic_loss.backward()
        actor_loss.backward()
        for param in itertools.chain(self.actor_model.parameters(),self.critic_model.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        self.global_step += 1

        if self.tau < 1:
            # Update target (soft)
            for target, model in zip((self.target_critic_model, self.target_actor_model), (self.critic_model, self.actor_model)):
                target_state_dict = target.state_dict()
                for key in target_state_dict:
                    target_state_dict[key] = (1 - self.tau) * model.state_dict()[key] \
                                             + self.tau * target_state_dict[key]
        elif self.global_step % self.tau == 0:
            # Update target (hard)
            self.target_critic_model.load_state_dict(self.critic_model.state_dict())
            self.target_actor_model.load_state_dict(self.actor_model.state_dict())

        return critic_loss

    def forward(self, states):
        with torch.no_grad():
            action = self.actor_model(torch.from_numpy(states)).numpy()
            if states.shape[0] == 1:
                return action.reshape((self.action_size, ))

            return action.reshape((states.shape[0], self.action_size))
