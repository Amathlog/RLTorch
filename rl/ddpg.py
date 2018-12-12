import itertools
from copy import copy

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

def format_file(file):
    parent = file.parent
    name = file.stem
    suffix = file.suffix

    critic_name = name + '_critic' + suffix
    actor_name = name + '_actor' + suffix

    return parent / critic_name, parent / actor_name


class DDPG(object):
    def __init__(self, state_size, action_size, critic_model, actor_model, learning_rate=(1e-5, 1e-4), gamma=0.95,
                 tau=2500, use_tensorboard=False, summary_location='.'):
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

        self.use_tensorboard = use_tensorboard
        self.writer = SummaryWriter(log_dir=summary_location) if use_tensorboard else None

        self.global_step = 0

        self.target_critic_model = copy(self.critic_model)
        self.target_actor_model = copy(self.actor_model)

        self.reset_targets()

        self.criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(list(self.critic_model.parameters()), lr=self.learning_rate[0])
        self.actor_optimizer = optim.Adam(list(self.actor_model.parameters()), lr=self.learning_rate[1])

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

        critic_loss = self.criterion(q_values, target)

        critic_loss.backward()

        for param in self.critic_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()

        actions_from_actor = self.actor_model(states)
        actor_loss = -self.critic_model(torch.cat((states, actions_from_actor), -1))
        actor_loss = actor_loss.mean()

        actor_loss.backward()
        for param in self.actor_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        if self.use_tensorboard and self.global_step % 100 == 0:
            self.writer.add_histogram("others/critic_output", q_values.detach().numpy(), self.global_step)
            self.writer.add_histogram("others/rewards", rewards.numpy(), self.global_step)
            self.writer.add_scalar('critic_loss', critic_loss.detach().numpy(), self.global_step)
            self.writer.add_scalar('actor_loss', actor_loss.detach().numpy(), self.global_step)
            for i in range(actions_from_actor.size()[1]):
                self.writer.add_histogram(f"actor_output/actor_output_{i}", actions_from_actor.detach().numpy()[:, i], self.global_step)
                self.writer.add_histogram(f"batch_action/batch_action_{i}", actions.numpy()[:, i], self.global_step)
            for i in range(states.size()[1]):
                self.writer.add_histogram(f"batch_input/batch_input_{i}", states.numpy()[:, i], self.global_step)

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

    def reset_targets(self):
        self.target_critic_model.load_state_dict(self.critic_model.state_dict())
        self.target_critic_model.eval()

        self.target_actor_model.load_state_dict(self.actor_model.state_dict())
        self.target_actor_model.eval()

    def save(self, file):
        critic_file, actor_file = format_file(file)
        self.critic_model.save(critic_file)
        self.actor_model.save(actor_file)

    def load(self, file, reset_target=True):
        critic_file, actor_file = format_file(file)
        self.actor_model.load(actor_file)
        self.critic_model.load(critic_file)

        if reset_target:
            self.reset_targets()
