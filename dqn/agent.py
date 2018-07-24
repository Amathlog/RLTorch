from dqn import DQN
from memory import Memory

import numpy as np
import torch

class Agent(object):
    def __init__(self, env, model, max_size_memory=100000):
        self.env = env
        self.model = model

        self.dqn = DQN(self.env.observation_space.shape[0],
                       self.env.action_space.n,
                       self.model, tau=1000)
        self.memory = Memory(max_size_memory)
        self.batch_size = 32

    def prepare_next_batch(self):
        batch = self.memory.sample(self.batch_size)
        # batch = np.array(batch, dtype=np.float32)

        with torch.no_grad():
            states = torch.from_numpy(np.array([b[0] for b in batch], dtype=np.float32))
            actions = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.float32))
            rewards = torch.from_numpy(np.array([b[2] for b in batch], dtype=np.float32))
            next_states = torch.from_numpy(np.array([b[3] for b in batch], dtype=np.float32))
            dones = torch.from_numpy(np.array([b[4] for b in batch], dtype=np.float32))

        return states, actions, rewards, next_states, dones

    def step(self, observation, current_step, warm_up=0, train=True):
        action = self.dqn.forward(torch.from_numpy(np.array([observation], dtype=np.float32))).numpy()
        next_observation, reward, done, info = self.env.step(action)
        loss = 0
        if train:
            self.memory.store([observation, action, reward, next_observation, 1 if done else 0])
            if current_step >= warm_up:
                loss = self.dqn.backward(*self.prepare_next_batch())

        return next_observation, reward, done, info, loss

    def fit(self, nb_episodes, warm_up=1000, verbose=True, episode_verbose=10):
        nb_steps = 0
        mean_reward = 0
        mean_loss = 0
        for episode in range(nb_episodes):
            current_reward = 0
            current_loss = 0
            current_steps = 0
            observation = self.env.reset()
            done = False
            while not done:
                next_observation, reward, done, info, loss = self.step(observation, nb_steps, warm_up, True)

                current_reward += reward
                current_loss += loss
                current_steps += 1

                nb_steps += 1

                if done:
                    break

            mean_reward += current_reward
            mean_loss += current_loss / current_steps
            if verbose and episode % episode_verbose == 0:
                print(f'Episode {episode}: Mean reward->{mean_reward / episode_verbose}; '
                      f'Mean Loss->{mean_loss / episode_verbose}')
                mean_reward = 0

    def test(self, nb_episodes, render=True, verbose=True):
        nb_steps = 0
        for episode in range(nb_episodes):
            current_reward = 0
            observation = self.env.reset()
            if render:
                self.env.render()
            done = False
            while not done:
                next_observation, reward, done, info, _ = self.step(observation, nb_steps, train=False)

                if render:
                    self.env.render()

                current_reward += reward

                nb_steps += 1

                if done:
                    break

            if verbose:
                print(f'Episode {episode}: Reward->{current_reward}')

if __name__ == "__main__":
    import gym
    from dqn import MLP
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = MLP([state_size, 32, 32, action_size])
    agent = Agent(env, model)

    agent.fit(1000, episode_verbose=100)

    agent.test(10)
