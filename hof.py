import random
import tensorflow as tf
from matplotlib import pyplot
import os
from math import floor
from model import Model


class HOF:
    def __init__(self, folder):
        self.hof = []
        self.folder = folder
        self.sample_history = []
        self.pop_size = 0
        self.basel = 0  # Used in limit-uniform sampling
        if not os.path.isdir(folder):
            os.makedirs(folder)

    def store(self, model):
        model.save_to("{}/{}".format(self.folder, self.pop_size))
        self.hof.append(self.pop_size)
        self.pop_size += 1
        self.basel += 1/self.pop_size**2

    # Gating method decides whether to add the model to the hall of fame
    def gate(self, model):
        # Simple gating method, stores model after every training episode
        self.store(model)

    # Samples from the hall of fame with the provided method
    def sample(self, method='uniform'):
        if method == 'limit-uniform':  # Performs poorly. Do not use.
            threshold = random.random()*self.basel

            cum_prob = 0
            ind = self.pop_size-1
            for i in range(self.pop_size):
                cum_prob += 1/(self.pop_size-i)**2
                if cum_prob > threshold:
                    ind = i
                    break
        elif method == 'uniform':
            ind = floor(random.random()*self.pop_size)

        self.sample_history.append(ind)

        name = self.hof[ind]
        return Model("{}/{}".format(self.folder, name))

    # Displays a histogram of the model iterations sampled from the hall of fame
    def sample_histogram(self, num=100):
        pyplot.hist(self.sample_history, num)
        pyplot.title("Sampling of Model Indices from HOF")
        pyplot.show()

