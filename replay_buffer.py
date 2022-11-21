import numpy as np
import random
import math

class PrioritySumTree():
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0

        self.depth = math.ceil(math.log2(capacity))
        self.array_len = 2**(self.depth + 1) - 1
        self.leaf_start = 2**self.depth - 1

        self.priorities = np.zeros(self.array_len, dtype="float32")
        self.priorities_min = np.full(shape=self.array_len, fill_value=np.Inf, dtype="float32")

        self.datatable = [None for _ in range(self.array_len)]
        self.ind = 0
        self.min = None

    def clear(self):
        self.priorities = np.zeros(self.array_len, dtype="float32")
        self.priorities_min = np.full(shape=self.array_len, fill_value=np.Inf, dtype="float32")
        self.datatable = [None for _ in range(self.array_len)]

        self.size = 0
        self.ind = 0

    def add(self, data, priority):
        self.size = min(self.size + 1, self.capacity)

        index = self.ind + self.leaf_start
        change = priority - self.priorities[index]

        self.priorities[index] = priority
        self.priorities_min[index] = priority
        self.datatable[index] = data

        current = (index - 1) // 2
        while (current >= 0):
            self.priorities[current] = self.priorities[current * 2 + 1] + self.priorities[current * 2 + 2]
            self.priorities_min[current] = min(self.priorities_min[current * 2 + 1], self.priorities_min[current * 2 + 2])
            current = (current - 1) // 2

        self.ind = (self.ind + 1) % self.capacity

    def update_vectorized(self, batch_ancestors, old_priorities, new_priorities):
        self.priorities[batch_ancestors[:, -1]] = self.priorities_min[batch_ancestors[:, -1]] = new_priorities

        # change = new_priorities - old_priorities
        # if change.shape == ():
            # change = np.expand_dims(change, axis=0)

        for i in range(batch_ancestors.shape[0]):
            for index in batch_ancestors[i][:-1:-1]:
                # self.priorities[batch_ancestors[i]] += change[i]
                self.priorities[index] = self.priorities[index * 2 + 1] + self.priorities[index * 2 + 2]
                self.priorities_min[index] = min(self.priorities_min[index * 2 + 1], self.priorities_min[index * 2 + 2])

    def get_total(self):
        return self.priorities[0]

    def get_min(self):
        return self.priorities_min[0]

    def sample_priority(self, val, record_ancestors=None):
        current = 0
        index = 0

        while (current < self.leaf_start):
            if record_ancestors is not None:
                record_ancestors[index] = current

            if val > self.priorities[2 * current + 1] and self.priorities[2 * current + 2] != 0:
                val -= self.priorities[2 * current + 1]
                current = 2 * current + 2
            else:
                if val > self.priorities[2 * current + 1]:
                    print("Avoided 0 priority node.")

                current = 2 * current + 1

            index += 1

        if record_ancestors is not None:
            record_ancestors[index] = current

        return self.datatable[current], self.priorities[current], self.get_min(), self.size


class ReplayBuffer:
    def __init__(self, capacity, batch_size, alpha):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = PrioritySumTree(capacity)
        self.index = 0

        self.last_batch = None
        self.last_batch_priorities = None
        self.max_priority = 2.0**alpha

    def clear(self):
        self.buffer.clear()

    def store(self, experience, priority=None):
        if priority is None:
            priority = self.max_priority
        self.buffer.add(experience, priority)

    def sample_batch(self):
        p_total = self.buffer.get_total()
        segment = p_total / self.batch_size

        experiences = []
        imp_sampling = np.zeros(self.batch_size, dtype="float32")

        self.last_batch = np.zeros(shape=(self.batch_size, self.buffer.depth + 1), dtype="int32")
        self.last_batch_priorities = np.zeros(self.batch_size, dtype="float32")

        for i in range(self.batch_size):
            priority = random.uniform(segment * i, segment * (i+1)) 
            experience, real_priority, min_priority, size = self.buffer.sample_priority(priority, self.last_batch[i])
            self.last_batch_priorities[i] = real_priority

            imp_sampling[i] = (1 / size) / (real_priority / p_total)
            imp_sampling[i] /= (1 / size) / (min_priority / p_total)
            experiences.append(experience)
            
        return experiences, imp_sampling

    def update_batch(self, priorities):
        assert self.last_batch is not None, "No batches have been sampled from this buffer."
        priorities = np.array(priorities)

        self.last_batch, unique_inds = np.unique(self.last_batch, axis=0, return_index=True)
        self.last_batch_priorities = self.last_batch_priorities[unique_inds]
        priorities = priorities[unique_inds]

        self.buffer.update_vectorized(self.last_batch, self.last_batch_priorities, priorities)

