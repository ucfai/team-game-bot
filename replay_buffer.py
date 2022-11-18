import numpy as np
import random
import math

class PrioritySumTree():
    def __init__(self, capacity):
        depth = math.ceil(math.log2(capacity))
        self.size = 2**(depth + 1) - 1
        self.leaf_start = 2**depth - 1

        self.arr = [[None, 0] for _ in range(self.size)]
        self.capacity = capacity
        self.ind = 0

    def clear(self):
        self.arr = [[None, 0] for _ in range(self.size)]
        self.ind = 0

    def add(self, data, priority):
        index = self.ind + self.leaf_start
        change = priority - self.arr[index][1]

        self.arr[index][1] = priority
        self.arr[index][0] = data

        current = (index - 1) // 2
        while (current >= 0):
            self.arr[current][1] += change
            current = (current - 1) // 2

        self.ind = (self.ind + 1) % self.capacity

    def update(self, index, priority):
        self.arr[index][1] = priority
        change = priority - self.arr[index][1]

        current = (index- 2) // 2
        while (current >= 0):
            self.arr[current][1] += change
            current = (current - 1) // 2

    def get_total(self):
        return self.arr[0][1]

    def sample_priority(self, val):
        current = 0

        while (current < self.leaf_start):
            if val > self.arr[2 * current + 1][1]:
                val -= self.arr[2 * current + 1][1]
                current = 2 * current + 2
            else:
                current = 2 * current + 1

        if self.arr[current][0] is None:
            print("oops ", current)

        return self.arr[current][0], current, (1 / self.capacity) * (self.arr[current][1] / self.get_total())


class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = PrioritySumTree(capacity)
        self.index = 0

        self.last_batch = None

    def clear(self):
        self.buffer.clear()

    def store(self, experience, priority=1e6):
        self.buffer.add(experience, priority)

    def sample_batch(self):
        p_total = self.buffer.get_total()
        segment = p_total / self.batch_size

        experiences = []
        indices = np.zeros(self.batch_size, dtype="int32")
        imp_sampling = np.zeros(self.batch_size)

        for i in range(self.batch_size):
            priority = random.uniform(segment * i, segment * (i+1)) 
            experience, indices[i], imp_sampling[i] = self.buffer.sample_priority(priority)
            experiences.append(experience)
            
        self.last_batch = indices
        return experiences, imp_sampling

    def update_batch(self, priorities):
        assert self.last_batch is not None, "No batches have been sampled from this buffer."

        for ind, priority in zip(self.last_batch, priorities):
            self.buffer.update(ind, priority)





