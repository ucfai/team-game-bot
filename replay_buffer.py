import random


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []

    def store(self, experience):
        self.buffer.append(experience)

        if len(self.buffer) > self.capacity:
            del self.buffer[0]

    def sample(self):
        if len(self.buffer) < self.batch_size:
            return self.buffer

        return random.sample(self.buffer, self.batch_size)



