import datetime
from pathlib import Path

import numpy as np

from rl.dqn import DQN
from rl.ddpg import DDPG
from rl.memory import Memory

class Agent(object):
    def __init__(self, env, model, policy, test_policy, batch_size=32, max_size_memory=100000, action_repetition=1,
                 save_folder=None):
        self.env = env
        self.model = model

        self.memory = Memory(max_size_memory)
        self.batch_size = batch_size
        self.policy = policy
        self.test_policy = test_policy

        self.action_repetition = action_repetition

        self.observation_shape = self.env.observation_space.shape

        self.best_reward = None
        self.reward_improved = False
        self.save_folder = save_folder

    def prepare_next_batch(self):
        batch = self.memory.sample(self.batch_size)

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.float32)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def step(self, observation, train=True):
        action = self.model.forward(np.array([observation], dtype=np.float32))

        action = self.policy(action) if train else self.test_policy(action)

        total_reward = 0
        next_observation = np.zeros(self.observation_shape)
        done = False
        info = {}
        steps = 0
        for _ in range(self.action_repetition):
            next_observation, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        reward = total_reward / steps
        next_observation = np.reshape(next_observation, self.observation_shape)

        if train:
            if hasattr(self.model, 'train'):
                self.memory.store([observation, action, reward, next_observation, 0 if done else 1])
            else:
                self.memory.store([observation, action, reward, next_observation, 1 if done else 0])

        return next_observation, reward, done, info

    def fit(self, nb_episodes, warm_up=1000, verbose=True, episode_verbose=10, test_period=None, render_test=False):
        self.policy.reset()
        nb_steps = 0
        mean_reward = 0
        mean_loss = 0
        for episode in range(1, nb_episodes + 1):
            if test_period is not None and episode % test_period == 0:
                self.test(3, render=render_test)
                if self.reward_improved:
                    if verbose:
                        print(f'Reward has improved: {self.best_reward}')
                    if self.save_folder is not None:
                        file = self.save_folder / (datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") +
                               f'_R={self.best_reward:.1f}.model')
                        if verbose:
                            print(f'Saving the model: {str(file)}')
                        self.model.save(file)

            current_reward = 0
            current_loss = 0
            current_steps = 0
            observation = self.env.reset()
            done = False
            while not done:
                next_observation, reward, done, info = self.step(observation, True)

                current_reward += reward
                current_steps += 1

                observation = next_observation

                nb_steps += 1

                if done:
                    break

            # Do the training only at the end of an episode
            if nb_steps >= warm_up:
                for _ in range(self.action_repetition * current_steps):
                    if hasattr(self.model, 'train'):
                        current_loss += self.model.backward(self.memory.sample(self.batch_size))[0]
                    else:
                        current_loss += self.model.backward(*self.prepare_next_batch())

            mean_reward += current_reward
            mean_loss += current_loss / current_steps
            if verbose and episode % episode_verbose == 0:
                print(f'[Train]Episode {episode}: Mean reward->{mean_reward / episode_verbose}; '
                      f'Mean Loss->{mean_loss / episode_verbose}')
                mean_reward = 0
                mean_loss = 0

    def test(self, nb_episodes, render=True, verbose=True, record_frames=False):
        self.test_policy.reset()
        nb_steps = 0
        frames = []
        full_reward = 0
        for episode in range(1, nb_episodes + 1):
            current_reward = 0
            observation = self.env.reset()
            if render:
                self.env.render()
            done = False
            while not done:
                next_observation, reward, done, info = self.step(observation, train=False)

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
                print(f'[Test]Episode {episode}: Reward->{current_reward}')
            full_reward += current_reward
        full_reward /= nb_episodes
        if self.best_reward is None or full_reward >= self.best_reward:
            self.best_reward = full_reward
            self.reward_improved = True
        else:
            self.reward_improved = False

        self.env.close()
        if self.env.env.viewer is not None:
            self.env.env.viewer = None
        return frames

if __name__ == "__main__":
    import sys

    import gym
    import car_env

    from rl.dqn import MLP
    from rl.policy import GreedyPolicy, AnnealingEpsilonGreedy, OUNoiseContinuousPolicy, NormalNoiseContinuousPolicy
    from DistributedRL.models.DDPG import DDPG as DDPG_Distrib

    ddpg = True
    pendulum = False
    train = False

    # model_to_load = None
    model_to_load = Path(r'C:\Users\alogut\Documents\Programming\RLTorch\rl\savedata\CarRacing') / '2018_09_14_12_42_R=149.5.model'

    if not ddpg:
        env = gym.make('CartPole-v0')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        model = MLP([state_size, 32, 32, action_size])
        algo = DQN(env.observation_space.shape[0],
                       env.action_space.n,
                       model, tau=1000, use_tensorboard=True)
        policy = AnnealingEpsilonGreedy(0.9, env.action_space, seed=0)
        test_policy = GreedyPolicy()
        env_name = 'CartPole'
    else:
        if pendulum:
            env = gym.make('Pendulum-v0')
            env_name = 'Pendulum'
        else:
            env = gym.make('CarRacingInternalState-v1')
            env_name = 'CarRacing'
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        critic_model = MLP([state_size + action_size, 256, 128, 64, 1])
        actor_model = MLP([state_size, 256, 128, 64, action_size], use_tanh=True,
                          output_support=(env.action_space.low, env.action_space.high))
        algo = DDPG(state_size, action_size, critic_model, actor_model, use_tensorboard=True, gamma=0.9, tau=2500,
                    learning_rate=(0.001, 0.0001))

        # algo = DDPG_Distrib(1, state_size, action_size, True, "DDPG", update=2500, action_scaling=((-1.0,1.0), (0.0, 1.0), (0.0, 1.0)))
        # algo.forward = algo.act
        # algo.backward = algo.train
        if pendulum:
            policy = OUNoiseContinuousPolicy(np.array([0.0]), np.array([0.2]), np.array([0.15]), env.action_space)
        else:
            # policy = OUNoiseContinuousPolicy(np.array([0.0]*3), np.array([0.2]*3), np.array([0.15, 0.15, -0.1]), env.action_space)
            policy = NormalNoiseContinuousPolicy(np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.2, 0.2]), env.action_space)
        test_policy = GreedyPolicy()

    action_repetition = 3 if ddpg and not pendulum else 1
    if model_to_load is not None:
        algo.load(model_to_load, reset_target=True)
    save_folder = Path('savedata') / f'{env_name}'
    if not save_folder.exists():
        save_folder.mkdir(parents=True, exist_ok=True)
    agent = Agent(env, algo, policy, test_policy, action_repetition=action_repetition, save_folder=save_folder)

    if train:
        agent.fit(5000, episode_verbose=10, test_period=100, render_test=False)

    agent.test(10)

    sys.exit(env.close())
