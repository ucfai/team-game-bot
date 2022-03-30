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
    def __init__(self, mnk, location=None):
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

        opt = SGD(learning_rate=0.02, momentum=0.0)

        self.model = Sequential()
        self.model.add(Conv2D(8, 3, activation='relu', padding="same", input_shape=(m, n, 2)))
        self.model.add(Conv2D(8, 3, activation='relu', padding="same"))

        self.model.add(Flatten())
        self.model.add(Dense(8, kernel_initializer='normal', activation='relu', input_shape=(1, m * n * 2)))
        self.model.add(Dense(mnk[0] * mnk[1], kernel_initializer='normal', activation='tanh'))

        self.model.compile(loss='mean_squared_error', optimizer=opt)

    def retrieve(self, location):
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
        """Evaluates the state of the board and returns the advantage of the current player.
        Changes 1 to mean the supplied player is at advantage, -1 disadvantage.

        Args:
            board (Board): Board object to be evaluated.

        Returns:
            tf.Tensor(1,1): Value indicating the advantage of the current player.
        """

        if board.who_won() != 2:
            return tf.constant(player * board.who_won(), dtype="float32", shape=(1, 1))
        else:
            action_value_vector = self.action_values(board)
            legal_action_values = output_rep.get_legal_vals(board, action_value_vector)
            return max(legal_action_values.values())

    def action_values(self, board):
        """Evaluates the advantage that the current player would have if he makes a
        given move on the board. Returns the value of taking a move from the given
        board state. Changes 1 to mean the supplied player would be at advantage, -1
        disadvantage.

        Args:
            board (Board): Board object where to make the move.
            move ((int, int)): (x, y) coordinates of the move to be played.

        Returns:
            tf.Tensor(1,1): Value indicating the advantage the player who made the move
                would have after making the move.
        """

        return self.model(get_input_rep(board.get_board()))

    def scheduler(self, epoch, lr):
        """Returns an epsilon value as a function of the current epoch.
        As a function of the epoch number, it returns a decreasing epsilon value
        used in the Epsilon-Greedy Method.

        Args:
            epoch (int): Number of training epoch.
            lr (???): ??? (Is this for the decay?)

        Returns:
            double: Epsilon value. Probability of choosing to explore.
        """
        if epoch < 5000:
            return 0.02
        elif epoch < 15000:
            return 0.01
        elif epoch < 25000:
            return 0.002
        else:
            return 0.001

    def td_update(self, state, action, next_state):
        """Performs a temporal difference update of the model.

        Args:
            board (Board): Board representing the current state of the game.
            greedy_move ((int, int)): Move to be played. Defaults to None.
            terminal (bool, optional): True if the current state of the game is terminal,
                False otherwise. Defaults to False.
        """
        m, n, k = self.mnk

        callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)

        start_board = Board(*self.mnk, state=state)
        next_board = Board(*self.mnk, state=next_state)

        prev_output = self.action_values(start_board)
        target_output = np.zeros(shape=prev_output.shape, dtype='float32')

        for move in start_board.legal_moves():
            index = move[0] * m + move[1]
            target_output[0][index] = prev_output[0][index]

        target_output[0][action[0] * m + action[1]] = self.state_value(next_board, player=state[1])

        self.model.fit(get_input_rep(start_board.get_board()), target_output, batch_size=1, verbose=0, callbacks=[callback])
