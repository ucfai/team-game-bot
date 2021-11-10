import random
import tensorflow as tf
from matplotlib import pyplot
import os
from math import floor


class HOF:
    def __init__(self, folder):
        self.hof = []
        self.folder = folder
        self.sample_history = []
        self.pop_size = 0
        self.basel = 0  # used in limit-uniform sampling
        if not os.path.isdir(folder):
            os.makedirs(folder)

    def store(self, model, name):
        model.save("{}/{}".format(self.folder, name))
        self.hof.append(name)
        self.pop_size += 1
        self.basel += 1/self.pop_size**2

    def sample_hof(self, method='uniform'):
        if method == 'limit-uniform':
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
        return tf.keras.models.load_model("{}/{}".format(self.folder, name))

    def sample_hist(self, num=100):
        pyplot.hist(self.sample_history, num)
        pyplot.title("Sampling of Model Indices from HOF")
        pyplot.show()

