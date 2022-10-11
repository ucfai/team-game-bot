from tensorflow.keras.optimizers import Adam

import mnk
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from state_representation import get_input_rep
import output_representation as output_rep
from mnk import Board


class Model:
    def __init__(self, mnk, lr=0.001, location=None):
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

        opt = SGD(learning_rate=lr)

        self.model = Sequential()
        self.model.add(Flatten(input_shape=(m, n, 2)))
        self.model.add(Dense(24, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(24, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(mnk[0] * mnk[1], kernel_initializer='normal', activation='tanh'))

        self.model.compile(loss='mean_squared_error', optimizer=opt)

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

    def get_target(self, state, action, next_state):
        m, n, k = self.mnk

        # TODO: Is this actually necessary? Might be wasteful
        start_board = Board(*self.mnk, state=state)
        next_board = Board(*self.mnk, state=next_state)

        prev_output = self.action_values(start_board)

        # OPT 1: If this line is used, illegal actions will be ignored.
        target_output = np.copy(prev_output)

        # OPT 2: If this is used, illegal actions will be trained to have action value -1.
        # target_output = np.full(shape=prev_output.shape, fill_value=-1, dtype='float32')
        #
        # for move in start_board.legal_moves():
        #    index = move[0] * m + move[1]
        #    target_output[0][index] = prev_output[0][index]

        target_output[0][action[0] * n + action[1]] = self.state_value(next_board, player=state[1])
        return target_output

    # Performs training on a single sample
    def td_update(self, state, action, next_state):
        """Performs a temporal difference update of the model.

        Args:
            state: Board representing the previous state of the game.
            action: Move played after previous state.
            next_state: Next state of the game after action was taken.
        """
        target_output = self.get_target(state, action, next_state)

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self.model.fit(get_input_rep(state), target_output, batch_size=1, verbose=0, callbacks=[lr_scheduler])


def scheduler(epoch, lr):
    if lr > 0.0005:
        return lr * tf.math.exp(-0.00005)
    else:
        return lr
