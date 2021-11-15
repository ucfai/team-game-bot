import mnk
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam

class Model:

    def __init__(self, location=False):

        # If a location is provided, retrieve the model stored at that location
        if location != False:
            self.model = self.retrieve(location)
            return

        opt = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999)

        self.model = Sequential()
        self.model.add(Dense(27, input_shape=(1, 9), kernel_initializer='normal', activation='tanh'))
        self.model.add(Dense(27, kernel_initializer='normal', activation='tanh'))
        self.model.add(Dense(1, kernel_initializer='normal', activation='tanh'))

        self.model.compile(loss='mean_squared_error', optimizer=opt)

    def retrieve(self, location):
        return tf.keras.models.load_model(location)

    def save_to(self, location):
        self.model.save(location)

    # Values closer to 1 mean X advantage, -1 means O advantage
    def raw_value(self, board):
        if board.who_won() != 2:
            return tf.constant(board.who_won(), dtype="float32", shape=(1, 1))
        else:
            return self.model(board.get_board())

    # Changes 1 to mean the supplied player is at advantage, -1 disadvantage
    def state_value(self, board, player):
        return player * self.raw_value(board)

    # Returns the value of taking a move from the given board state
    def action_value(self, board, move):
        player = board.player

        board.move(*move)
        val = self.state_value(board, player)
        board.undo_move(*move)

        return val

    # Performs a temporal difference update of the model
    def td_update(self, board, greedy_move=(), terminal=False):
        if terminal:
            assert board.who_won() != 2
            assert greedy_move == ()
            self.model.fit(board.history()[-2], self.raw_value(board), batch_size=1, verbose=0)
        else:
            self.model.fit(board.history()[-2], self.action_value(board, greedy_move), batch_size=1, verbose=0)

