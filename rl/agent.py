from rl.dqn import DQN
from rl.ddpg import DDPG
from rl.memory import Memory

import numpy as np

class Agent(object):
    def __init__(self, env, model, policy, test_policy, batch_size=32, max_size_memory=100000, action_repetition=1):
        self.env = env
        self.model = model

        self.memory = Memory(max_size_memory)
        self.batch_size = batch_size
        self.policy = policy
        self.test_policy = test_policy

        self.action_repetition = action_repetition

        self.observation_shape = self.env.observation_space.shape


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

    def fit(self, nb_episodes, warm_up=1000, verbose=True, episode_verbose=10, test_render_period=None):
        self.policy.reset()
        nb_steps = 0
        mean_reward = 0
        mean_loss = 0
        for episode in range(1, nb_episodes + 1):
            if test_render_period is not None and episode % test_render_period == 0:
                self.test(3)
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
        self.env.close()
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
    else:
        if pendulum:
            env = gym.make('Pendulum-v0')
        else:
            env = gym.make('CarRacingInternalState-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        critic_model = MLP([state_size + action_size, 64, 64, 32, 16, 1])
        actor_model = MLP([state_size, 64, 64, 32, 16, action_size], use_tanh=True,
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
    agent = Agent(env, algo, policy, test_policy, action_repetition=action_repetition)

    agent.fit(1000, episode_verbose=10, test_render_period=100)

    agent.test(10)

    sys.exit(env.close())
