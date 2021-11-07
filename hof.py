import random
import tensorflow as tf
from math import floor
import os


class HOF:
    def __init__(self, folder):
        self.hof = []
        self.folder = folder
        if not os.path.isdir(folder):
            os.makedirs(folder)

    def store(self, model, name):
        model.save("{}/{}".format(self.folder, name))
        self.hof.append(name)

    def sample_hof(self):
        pop_size = len(self.hof)
        ind = floor(pop_size*random.random())
        name = self.hof[ind]
        return tf.keras.models.load_model("{}/{}".format(self.folder, name))

