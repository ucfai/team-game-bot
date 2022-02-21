import random
import tensorflow as tf
from matplotlib import pyplot
import os
from math import floor
from model import Model
from agent import Agent
from mnk import Board
from utils import run_game


class HOF:
    def __init__(self, mnk, folder):
        self.hof = []
        self.folder = folder
        self.mnk = mnk
        self.sample_history = []
        self.pop_size = 0
        self.basel = 0  # Used in limit-uniform sampling
        if not os.path.isdir(folder):
            os.makedirs(folder)

    def store(self, model):
        model.save_to("{}/{}".format(self.folder, self.pop_size))
        self.hof.append(self.pop_size)
        self.pop_size += 1
        self.basel += 1 / self.pop_size**2

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
        elif method == 'naive':
            ind = self.pop_size-1

        self.sample_history.append(ind)

        name = self.hof[ind]
        return Model("{}/{}".format(self.folder, name))

    ''' === MOVED TO PLOT.PY LMK IF I CAN DELETE IT FROM HERE ===
    # Displays a histogram of the model iterations sampled from the hall of fame
    def sample_histogram(self, num=100):
        pyplot.hist(self.sample_history, num)
        pyplot.title("Sampling of Model Indices from HOF")
        pyplot.show()
    '''

    ''' === MOVED TO PLOT.PY LMK IF I CAN DELETE IT FROM HERE ===
    # Displays a winrate matrix of the historical policies for the given player
    def winrate_matrix(self, iterations):
        matrix = []
        for i in range (0, self.pop_size, iterations):
            matrix.append([])
            for j in range (0, self.pop_size, iterations):
                model_i = Model("{}/{}".format(self.folder, self.hof[i]))
                model_j = Model("{}/{}".format(self.folder, self.hof[j]))

                value = run_game(Agent(model_i, 1), Agent(model_j, -1))[0]
                matrix[-1].append(value)

        pyplot.imshow(matrix, cmap="bwr")
        pyplot.imsave("plots/Matrix.png", matrix, cmap="bwr")
    '''