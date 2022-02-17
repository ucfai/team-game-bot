import mnk
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from state_representation import get_input_rep


class Model:
    def __init__(self, location=None):
        """Tic-Tac-Toe Game Evaluator Model.
        Provides a Convolutional Neural Network that can be trained to evaluate different
        board states, determining which player has the advantage at any given state. 

        Args:
            location (str, optional): Path to where the model is located. If none
                is provided a new model is initialized. Defaults to None.
        """

        # If a location is provided, retrieve the model stored at that location
        if location is not None:
            self.model = self.retrieve(location)
            return

        opt = SGD(learning_rate=0.02, momentum=0.0)

        self.model = Sequential()
        self.model.add(Conv2D(48, 3, activation='relu', input_shape=(3, 3, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(27, kernel_initializer='normal', activation='relu', input_shape=(1, 18)))
        self.model.add(Dense(1, kernel_initializer='normal', activation='tanh'))

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

    def raw_value(self, board):
        """Evaluates the players advantage of given a board state.
        Given board state, the model evaluates and it returns a value in the range
        (-1, 1) indicating which player has the advantage at the current state.
        Values closer to 1 mean "X" advantage, -1 means "O" advantage.

        Args:
            board (Board): Board object to be evaluated.

        Returns:
            tf.Tensor(1,1): Value indicating which player has the advantage according
                to the model. (advantage): (-1) "O" <--...0...--> "X" (+1)
        """
        if board.who_won() != 2:
            return tf.constant(board.who_won(), dtype="float32", shape=(1, 1))
        else:
            return board.player*self.model(get_input_rep(board.get_board()))

    def raw_action_value(self, board, move):
        """Evaluates the players advantage if a given move was made on the board.
        Given a board state and a move to be played, the model evaluates the board that
        results from this move and returns a value in the range (-1, 1) indicating which
        player has the advantage after the move. Values closer to 1 mean "X" advantage,
        -1 means "O" advantage.

        Args:
            board (Board): Board object where to make the move.
            move ((int, int)): (x, y) coordinates of the move to be played.

        Returns:
            tf.Tensor(1,1): Value indicating which player has the advantage after the move
                according to the model. (advantage): (-1) "O" <--...0...--> "X" (+1)
        """
        board.move(*move)
        val = self.raw_value(board)
        board.undo_move(*move)

        return val


    def state_value(self, board):
        """Evaluates the state of the board and returns the advantage of the current player.
        Changes 1 to mean the supplied player is at advantage, -1 disadvantage.

        Args:
            board (Board): Board object to be evaluated.

        Returns:
            tf.Tensor(1,1): Value indicating the advantage of the current player.
        """
        if board.who_won() == 0:
            return tf.constant(0, dtype="float32", shape=(1, 1))
        elif board.who_won() == board.player:
            return tf.constant(1, dtype="float32", shape=(1, 1))
        elif board.who_won() == -1*board.player:
            return tf.constant(-1, dtype="float32", shape=(1, 1))
        else:
            return self.model(get_input_rep(board.get_board()))

    # 
    def action_value(self, board, move):
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
        board.move(*move)
        val = self.state_value(board)
        board.undo_move(*move)
        return val

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


    def td_update(self, board, greedy_move=None, terminal=False):
        """Performs a temporal difference update of the model.

        Args:
            board (Board): Board representing the current state of the game.
            greedy_move ((int, int)): Move to be played. Defaults to None.
            terminal (bool, optional): True if the current state of the game is terminal,
                False otherwise. Defaults to False.
        """
        # Ensures td_update is possible (agent has experienced 2 states)
        if len(board.history()) < 3:
            return

        callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        if terminal:
            assert board.who_won() != 2
            assert greedy_move is None
            self.model.fit(get_input_rep(board.history()[-2]), self.state_value(board), batch_size=1, verbose=0, callbacks=[callback])
        else:
            self.model.fit(get_input_rep(board.history()[-2]), self.action_value(board, greedy_move), batch_size=1, verbose=0, callbacks=[callback])

