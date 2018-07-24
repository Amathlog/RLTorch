from collections import deque
from random import sample

import numpy as np


class Memory(object):
    def __init__(self, max_size, file=None):
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        assert(len(self.memory) >= batch_size)
        return sample(self.memory, batch_size)

    def save(self, file):
        np.save(file, self.memory)

    def load(self, file):
        self.memory = deque(np.load(file).tolist(), maxlen=self.max_size)
