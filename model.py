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
        m, n, k = mnk

        # If a location is provided, retrieve the model stored at that location
        if location is not None:
            self.model = self.retrieve(location)
            return

        if model is not None:
            self.model = model
            return

        self.opt = Adam(learning_rate=lr)
        regularization = 0.0001

        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=3, input_shape=(m, n, 2), kernel_regularizer=l2(regularization)))
        self.model.add(Flatten())
        self.model.add(Dense(128, kernel_initializer='normal', activation='relu', kernel_regularizer=l2(regularization)))
        self.model.add(Dense(mnk[0] * mnk[1], kernel_initializer='normal', kernel_regularizer=l2(regularization)))

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

    def state_value(self, board, player):
        """Evaluates the state of the board and returns the advantage of the given player.
        1 means the supplied player is at advantage, -1 disadvantage.

        Args:
            board (Board): Board object to be evaluated.
            player: Player being used as point of reference.

        Returns:
            tf.Tensor(shape=(1,1)): Value indicating the advantage of the current player.
        """

        if board.who_won() != 2:
            return tf.constant(player * board.who_won(), dtype="float32", shape=(1, 1))
        else:
            action_value_vector = self.action_values(board)
            legal_action_values = output_rep.get_legal_vals(board, action_value_vector)
            return max(legal_action_values.values())

    def action_values(self, board):
        """Returns the vector of action values for all actions in the current board state. This includes
        illegal actions that cannot be taken.

        Args:
            board (Board): Board object representing current state.
        Returns:
            tf.Tensor(shape=(m * n)): Vector where entry i indicates the value of taking move i from the current state.
        """

        return self.model(get_input_rep(board.get_board()))


def scheduler(epoch, lr):
    if lr > 0.0001:
        return lr * tf.math.exp(-0.0009)
    else:
        return lr
