import random
import numpy as np

class Policy(object):
    def __call__(self, action):
        return self.act(action)

    def act(self, action):
        raise NotImplementedError

    def reset(self):
        pass


class GreedyPolicy(Policy):
    def act(self, action):
        return action


class EpsilonGreedy(Policy):
    def __init__(self, epsilon, action_space, seed=None):
        self.epsilon = epsilon
        self.action_space = action_space

        if seed is not None:
            random.seed(seed)

    def act(self, action):
        if random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        return action


class AnnealingEpsilonGreedy(EpsilonGreedy):
    def __init__(self, epsilon, action_space, annealing = 0.999, min_value = 0.1, seed=None):
        super().__init__(epsilon, action_space, seed)

        self.epsilon_init = epsilon

        self.annealing = annealing
        self.min_value = min_value

    def act(self, action):
        action = super().act(action)
        self.epsilon = max(self.min_value, self.epsilon*self.annealing)
        return action

    def reset(self):
        self.epsilon = self.epsilon_init

class NormalNoiseContinuousPolicy(Policy):
    def __init__(self, mus, sigmas, action_space, seed=None):
        self.mus = mus
        self.sigmas = sigmas
        self.action_space = action_space

        assert self.mus.shape == self.sigmas.shape == self.action_space.shape

        if seed is not None:
            np.random.seed(seed)

    def act(self, action):
        noised_action = action + np.random.normal(self.mus, self.sigmas)
        return np.clip(noised_action, self.action_space.low, self.action_space.high)