import random


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.index = 0

    def clear(self):
        self.buffer = []
        self.index = 0

    def store(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer[self.index] = experience
            self.index = (self.index + 1) % self.capacity
        else:
            self.buffer.append(experience)

    def sample(self):
        if len(self.buffer) < self.batch_size:
            return self.buffer

        return random.sample(self.buffer, self.batch_size)



