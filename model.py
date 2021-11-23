import mnk
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD

class Model:

    def __init__(self, location=False):

        # If a location is provided, retrieve the model stored at that location
        if location != False:
            self.model = self.retrieve(location)
            return

        opt = SGD(learning_rate=0.02, momentum=0.0)

        self.model = Sequential()
        self.model.add(Conv2D(48, 3, activation='relu', input_shape=(3,3,2)))
        self.model.add(Flatten())
        self.model.add(Dense(27, kernel_initializer='normal', activation='relu', input_shape=(1,18)))
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
            return board.player*self.model(board.get_board())

    def raw_action_value(self, board, move):
        board.move(*move)
        val = self.raw_value(board)
        board.undo_move(*move)

        return val

    # Changes 1 to mean the supplied player is at advantage, -1 disadvantage
    def state_value(self, board):
        if board.who_won() == 0:
            return tf.constant(0, dtype="float32", shape=(1, 1))
        elif board.who_won() == board.player:
            return tf.constant(1, dtype="float32", shape=(1, 1))
        elif board.who_won() == -1*board.player:
            return tf.constant(-1, dtype="float32", shape=(1, 1))
        else:
            return self.model(board.get_board())

    # Returns the value of taking a move from the given board state
    def action_value(self, board, move):
        board.move(*move)
        val = self.state_value(board)
        board.undo_move(*move)

        return val

    def scheduler(self, epoch, lr):
        if epoch < 5000:
            return 0.02
        elif epoch < 15000:
            return 0.01
        elif epoch < 25000:
            return 0.002
        else:
            return 0.001

    # Performs a temporal difference update of the model
    def td_update(self, board, greedy_move=(), terminal=False):
        callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        if terminal:
            assert board.who_won() != 2
            assert greedy_move == ()
            self.model.fit(board.history()[-2], self.state_value(board), batch_size=1, verbose=0, callbacks=[callback])
        else:
            self.model.fit(board.history()[-2], self.action_value(board, greedy_move), batch_size=1, verbose=0, callbacks=[callback])

