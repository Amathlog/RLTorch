import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import copy

class MLP(nn.Module):
    """
    Create a MLP given layer sizes (input, hidden_layers, output)
    Use setattr and getattr because if we store layers in a list, PyTorch won't find them.
    """
    def __init__(self, layers_sizes, use_tanh=False):
        super().__init__()

        self.nb_layers = 0
        self.use_tanh = use_tanh
        for i in range(len(layers_sizes) - 1):
            self.add_layer(i, nn.Linear(layers_sizes[i], layers_sizes[i + 1]))

    def add_layer(self, idx, layer):
        setattr(self, f'layer_{idx}', layer)
        self.nb_layers += 1

    def get_layer(self, idx):
        if idx < 0:
            # Going backwards
            idx = self.nb_layers + idx
        assert 0 <= idx < self.nb_layers, 'Tried to access a layer that does not exists'
        return getattr(self, f'layer_{idx}')

    def forward(self, x):
        for i in range(self.nb_layers-1):
            x = F.relu(self.get_layer(i)(x))
        x = self.get_layer(-1)(x)
        if self.use_tanh:
            return F.tanh(x)
        return x


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
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(list(self.model.parameters()), lr=self.learning_rate, momentum=0.9)

    def backward(self, states, actions, rewards, next_states, dones):

        # Convert numpy arrays to Tensors
        with torch.no_grad():
            states, actions, rewards, next_states, dones = \
                map(torch.from_numpy, [states, actions, rewards, next_states, dones])

        q_values = self.model(states).gather(1, actions.long().view(-1, 1)).view(-1)
        # Bellman target
        next_q_values = torch.max(self.target_model(next_states), dim=-1)[0].detach()
        target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.global_step += 1

        if self.tau < 1:
            # Update target (soft)
            # Not sure it is working
            target_state_dict = self.target_model.state_dict()
            for key in target_state_dict:
                target_state_dict[key] = (1 - self.tau) * self.model.state_dict()[key] \
                                         + self.tau * target_state_dict[key]
        elif self.global_step % self.tau == 0:
            # Update target (hard)
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.detach().numpy()

    def forward(self, states):
        with torch.no_grad():
            states = torch.from_numpy(states)
            q_values = self.model(states)
            best_action = torch.argmax(q_values)
            return best_action.numpy()
