from tensorflow.keras.optimizers import Adam

import mnk
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from state_representation import get_input_rep
import output_representation as output_rep
from mnk import Board


class Model:
    def __init__(self, mnk, lr=0.001, location=None, model=None):
        """Tic-Tac-Toe Game Evaluator Model.
        Provides a Convolutional Neural Network that can be trained to evaluate different
        board states, determining which player has the advantage at any given state. 

        Args:
            location (str, optional): Path to where the model is located. If none
                is provided a new model is initialized. Defaults to None.
        """
        self.mnk = mnk
        self.lr = lr

        # If a location is provided, retrieve the model stored at that location
        if location is not None:
            self.model = self.retrieve(location)
            return
        elif model is not None:
            self.model = model
            return
        else:
            self.initialize_model()

    def reset_optimizer(self):
        self.opt = Adam(learning_rate=self.lr)
        self.model.compile(loss='mean_squared_error', optimizer=self.opt)

    def initialize_model(self, regularization=0.0001):
        m, n, k = self.mnk

        self.model = Sequential()
        #self.model.add(Conv2D(filters=16, kernel_size=3, padding="same", input_shape=(m, n, 2), kernel_regularizer=l2(regularization)))
        #self.model.add(Conv2D(filters=32, kernel_size=3, padding="same", kernel_regularizer=l2(regularization)))
        #self.model.add(Conv2D(filters=16, kernel_size=3, padding="same", input_shape=(m, n, 2), kernel_regularizer=l2(regularization)))
        #self.model.add(Conv2D(filters=1, kernel_size=3, padding="same", kernel_regularizer=l2(regularization)))
        #self.model.add(Flatten())

        self.model.add(Conv2D(filters=32, kernel_size=3, input_shape=(m, n, 2), kernel_regularizer=l2(regularization)))
        self.model.add(Flatten())
        self.model.add(Dense(128, kernel_initializer='normal', activation='relu', kernel_regularizer=l2(regularization)))
        self.model.add(Dense(m * n, kernel_initializer='normal', kernel_regularizer=l2(regularization)))

        self.opt = Adam(learning_rate=self.lr)
        self.model.compile(loss='mean_squared_error', optimizer=self.opt)

    @staticmethod
    def retrieve(location):
        """Retrieves keras model located at the given path and returns it.

        Args:
            location (str): Path to where the model is located.

        Returns:
            keras.Model: Model retrieved from path.
        """
        return tf.keras.models.load_model(location)

    def save_to(self, location):
        """Saves the current model at the given directory path.

        Args:
            location (str): Directory path where the model will be saved.
        """
        self.model.save(location)

    def state_value(self, states, terminal=None):
        """Evaluates the state of the board and returns the advantage of the given player.
        1 means the supplied player is at advantage, -1 disadvantage.

        Args:
            board (Board): Board object to be evaluated.
            player: Player being used as point of reference.

        Returns:
            tf.Tensor(shape=(1,1)): Value indicating the advantage of the current player.
        """

        action_vals = self.model(states)
        k, m, n, _ = states.shape

        illegal_actions = (np.sum(states, axis=3) != 0).reshape(k, m * n)
        
        # Replace values for illegal actions with -infinity so they can't be picked as max
        action_vals = np.where(illegal_actions, np.full(shape=(k, m * n), fill_value=np.NINF, dtype="float32"), action_vals)
        max_vals = tf.math.reduce_max(action_vals, axis=1)

        if terminal is not None:
            # If state is terminal, return an index of -1 for that state
            max_inds = np.where(terminal, np.full(shape=k, fill_value=-1, dtype="int32"), np.argmax(action_vals, axis=1))
        else:
            max_inds = np.argmax(action_vals, axis=1)

        return max_vals, max_inds

    def action_values(self, states):
        """Returns the vector of action values for all actions in the current board state. This includes
        illegal actions that cannot be taken.

        Args:
            board (Board): Board object representing current state.
        Returns:
            tf.Tensor(shape=(m * n)): Vector where entry i indicates the value of taking move i from the current state.
        """

        return self.model(states)


def scheduler(epoch, lr):
    if lr > 0.0001:
        return lr * tf.math.exp(-0.0009)
    else:
        return lr
