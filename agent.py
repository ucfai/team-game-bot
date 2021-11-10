import mnk
import keras.models
import tensorflow as tf
import random


class Agent:

    def __init__(self, board, model, player, training):
        self.board = board
        self.model = model
        self.player = player
        self.training = training

    def greedy_action(self):
        legal_moves = self.board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        best_move = legal_moves[-1]
        max_evaluation = -1

        for move in legal_moves:
            self.board.move(*move)

            val = self.value()
            if val > max_evaluation:
                best_move = move
                max_evaluation = val

            self.board.undo_move(*move)

        return best_move

    def random_action(self):
        legal_moves = self.board.legal_moves()
        return legal_moves[random.randint(0, len(legal_moves) - 1)]

    def value(self):
        if self.board.who_won() == self.player:
            return tf.constant(1, dtype="float32", shape=(1, 1))
        elif self.board.who_won() == -1*self.player:
            return tf.constant(-1, dtype="float32", shape=(1, 1))
        elif self.board.who_won() == 0:
            return tf.constant(0, dtype="float32", shape=(1, 1))
        else:
            return self.player*self.model(self.board.get_board())

    def action(self, epsilon=0):
        legal_moves = self.board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        greedy = self.greedy_action()
        if self.training and len(self.board.history()) >= (2 + (self.player == -1)):
            self.update_model(greedy)

        # Exploration
        if random.random() < epsilon:
            print("Played epsilon move ({:.5f})".format(epsilon))
            move = self.random_action()
        else:
            move = greedy

        self.board.move(*move)

    def update_model(self, greedy_move=()):
        if greedy_move == ():
            assert self.board.who_won() != 2 and self.board.who_won() != self.player
            self.model.fit(self.board.history()[-2], self.value(), batch_size=1, verbose=0)
        else:
            self.board.move(*greedy_move)
            self.model.fit(self.board.history()[-3], self.value(), batch_size=1, verbose=0)
            self.board.undo_move(*greedy_move)



