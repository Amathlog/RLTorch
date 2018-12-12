from enum import Enum
import logging
import random

import numpy as np


class State(Enum):
    RESET = 0
    CHOOSE_ACTION = 1
    DO_ACTION = 2
    UPDATE_Q = 3


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

def action_to_numpy(a):
    if a == Action.UP:
        return np.array([-1, 0])
    if a == Action.RIGHT:
        return np.array([0, 1])
    if a == Action.DOWN:
        return np.array([1, 0])
    if a == Action.LEFT:
        return np.array([0, -1])

    raise ValueError()


NB_ACTIONS = len(Action)


class Grid(object):
    def __init__(self, grid_file):
        self.load_file(grid_file)
        self.reset()

        self.action_reward = -0.1
        self.win_reward = 1
        self.loss_reward = -1

    def load_file(self, grid_file):
        with grid_file.open('r') as f:
            data = f.readlines()
        self.height = len(data)
        self.width = len(data[0]) - 1

        self.data = []

        for i, line in enumerate(data):
            self.data.append([])
            for j, c in enumerate(line):
                if c == 'S':
                    self.start = (i, j)
                elif c == '\n':
                    continue
                self.data[-1].append(c)

        self.data = np.array(self.data)

    def reset(self):
        self.current_pos = np.array(self.start)
        self.previous_pos = np.array(self.current_pos)

    def step(self, action):
        """
        Do a step given the action
        :return: A tuple (reward, is_terminal)
        """
        next_pos = self.current_pos + action_to_numpy(action)

        # Check if we got out of the grid or next pos is a wall. If yes, do nothing.
        if next_pos[0] < 0 or next_pos[0] >= self.height or \
            next_pos[1] < 0 or next_pos[1] >= self.width or \
            self.data[tuple(next_pos)] == 'X':
            return self.action_reward, False

        self.previous_pos = np.array(self.current_pos)
        self.current_pos = next_pos
        # Check if we arrived at a terminal state
        if self.data[tuple(self.current_pos)] != '.':
            return self.win_reward if self.data[tuple(self.current_pos)] == 'W' else self.loss_reward, True

        return self.action_reward, False

    def __repr__(self):
        res = ""
        for i in range(self.height):
            for j in range(self.width):
                if np.all((i,j) == self.current_pos):
                    res += 'o'
                else:
                    res += self.data[i, j]
            res += '\n'
        return res


class QLearning(object):

    def __init__(self, grid):
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.grid = grid
        self.epoch = 0
        self.current_state = State.RESET

        self.is_terminal = False
        self.current_action = None
        self.previous_pos = None
        self.reward = None

        self.tab = np.zeros((grid.height, grid.width, NB_ACTIONS))

        self.logger = logging.getLogger("QLearning")
        logging.basicConfig(level=logging.WARNING)

    def choose_action(self, greedy=False, pos=None):
        if random.uniform(0, 1) < self.epsilon and not greedy:
            self.logger.debug(f"Action chosen randomly")
            return Action(random.randint(0, NB_ACTIONS - 1))

        if pos is None:
            pos = grid.current_pos

        self.logger.debug(f'QValues: {self.tab[tuple(pos)]}')

        return Action(np.argmax(self.tab[tuple(pos)]))

    def update_q(self):
        x, y = self.previous_pos
        x_next, y_next = grid.current_pos
        next_action = self.choose_action(True)
        current_q = self.tab[x, y, self.current_action.value]
        next_q = self.tab[x_next, y_next, next_action.value]

        update = current_q + self.alpha * (self.reward + self.gamma * next_q - current_q)

        self.logger.debug("Q(s,a) + alpha*(r + gamma* max_a'Q(s',a') - Q(s,a)")
        self.logger.debug(f"{current_q:.3f} + {self.alpha:.2f}*({self.reward:.1f} + "
                          f"{self.gamma:.2f} * {next_q:.3f} - {current_q:.3f}) = {update:.3f}")

        self.tab[x, y, self.current_action.value] = update

    def step(self):
        if self.current_state == State.RESET:
            self.logger.debug("Reset...")
            self.grid.reset()
            self.logger.debug('\n' + str(self.grid))
            self.previous_pos = grid.current_pos
            self.current_state = State.CHOOSE_ACTION
            return

        if self.current_state == State.CHOOSE_ACTION:
            self.logger.debug("Choose action...")
            self.current_action = self.choose_action()
            self.logger.debug(f"Action chosen: {self.current_action}")
            self.current_state = State.DO_ACTION
            return

        if self.current_state == State.DO_ACTION:
            self.logger.debug("Do action...")
            self.previous_pos = np.array(grid.current_pos)
            self.reward, self.is_terminal = self.grid.step(self.current_action)
            self.logger.debug('\n' + str(self.grid))
            self.current_state = State.UPDATE_Q
            return

        if self.current_state == State.UPDATE_Q:
            self.logger.debug("Update Q...")
            self.update_q()
            if self.is_terminal:
                self.current_state = State.RESET
                self.epoch += 1
            else:
                self.current_state = State.CHOOSE_ACTION
            return


if __name__ == "__main__":
    from pathlib import Path
    grid_path = Path('grid.txt')

    grid = Grid(grid_path)
    # print(grid)
    # print(grid.step(Action.RIGHT))
    # print(grid)
    # print(grid.step(Action.UP))
    # print(grid)
    # print(grid.step(Action.RIGHT))
    # print(grid)
    # print(grid.step(Action.UP))
    # print(grid)
    # print(grid.step(Action.UP))
    # print(grid)
    # print(grid.step(Action.RIGHT))
    # print(grid)
    q_learning = QLearning(grid)
    for i in range(1000):
        q_learning.step()