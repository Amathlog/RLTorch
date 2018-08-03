from rl.dqn import DQN
from rl.ddpg import DDPG
from rl.memory import Memory

import numpy as np

class Agent(object):
    def __init__(self, env, model, policy, test_policy, batch_size=32, max_size_memory=100000):
        self.env = env
        self.model = model

        self.memory = Memory(max_size_memory)
        self.batch_size = batch_size
        self.policy = policy
        self.test_policy = test_policy

        self.observation_shape = self.env.observation_space.shape


    def prepare_next_batch(self):
        batch = self.memory.sample(self.batch_size)

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.float32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def step(self, observation, current_step, warm_up=0, train=True):
        action = self.model.forward(np.array([observation], dtype=np.float32))

        action = self.policy(action) if train else self.test_policy(action)

        next_observation, reward, done, info = self.env.step(action)
        next_observation = np.reshape(next_observation, self.observation_shape)

        loss = 0
        if train:
            self.memory.store([observation, action, reward, next_observation, 1 if done else 0])
            if current_step >= warm_up:
                loss = self.model.backward(*self.prepare_next_batch())

        return next_observation, reward, done, info, loss

    def fit(self, nb_episodes, warm_up=1000, verbose=True, episode_verbose=10):
        self.policy.reset()
        nb_steps = 0
        mean_reward = 0
        mean_loss = 0
        for episode in range(1, nb_episodes + 1):
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

                observation = next_observation

                nb_steps += 1

                if done:
                    break

            mean_reward += current_reward
            mean_loss += current_loss / current_steps
            if verbose and episode % episode_verbose == 0:
                print(f'Episode {episode}: Mean reward->{mean_reward / episode_verbose}; '
                      f'Mean Loss->{mean_loss / episode_verbose}')
                mean_reward = 0
                mean_loss = 0

    def test(self, nb_episodes, render=True, verbose=True, record_frames=False):
        self.test_policy.reset()
        nb_steps = 0
        frames = []
        for episode in range(1, nb_episodes + 1):
            current_reward = 0
            observation = self.env.reset()
            if render:
                self.env.render()
            done = False
            while not done:
                next_observation, reward, done, info, _ = self.step(observation, nb_steps, train=False)

                if render:
                    if record_frames:
                        frames.append(self.env.render(mode='rgb_array'))
                    else:
                        self.env.render()

                current_reward += reward

                observation = next_observation

                nb_steps += 1

                if done:
                    break

            if verbose:
                print(f'Episode {episode}: Reward->{current_reward}')

        return frames

if __name__ == "__main__":
    import sys

    import gym

    from rl.dqn import MLP
    from rl.policy import GreedyPolicy, AnnealingEpsilonGreedy, NormalNoiseContinuousPolicy

    ddpg = True

    if not ddpg:
        env = gym.make('CartPole-v0')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        model = MLP([state_size, 32, 32, action_size])
        algo = DQN(env.observation_space.shape[0],
                       env.action_space.n,
                       model, tau=1000)
        policy = AnnealingEpsilonGreedy(0.9, env.action_space, seed=0)
        test_policy = GreedyPolicy()
    else:
        env = gym.make('Pendulum-v0')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        critic_model = MLP([state_size + action_size, 32, 32, 1])
        actor_model = MLP([state_size, 32, 32, action_size], use_tanh=True)
        algo = DDPG(state_size, action_size, critic_model, actor_model, action_support=(env.action_space.low, env.action_space.high))
        policy = NormalNoiseContinuousPolicy(np.array([0.0]), np.array([0.1]), env.action_space)
        test_policy = GreedyPolicy()

    agent = Agent(env, algo, policy, test_policy)

    agent.fit(20000, episode_verbose=100)

    agent.test(10)

    sys.exit(env.close())
