import mnk
import keras.models
import tensorflow as tf
import random


class Agent:

    def __init__(self, model, player):
        self.model = model
        self.player = player

    def greedy_action(self, board):
        legal_moves = board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        best_move = legal_moves[-1]
        max_evaluation = -1

        for move in legal_moves:
            board.move(*move)

            val = self.value(board)
            if val > max_evaluation:
                best_move = move
                max_evaluation = val

            board.undo_move(*move)

        return best_move

    def random_action(self, board):
        legal_moves = board.legal_moves()
        return legal_moves[random.randint(0, len(legal_moves) - 1)]

    def value(self, board):
        if board.who_won() != 2:
            return tf.constant(self.player*board.who_won(), dtype="float32", shape=(1, 1))
        else:
            return self.player*self.model(board.get_board())

    def evaluation(self, board):
        if board.who_won() != 2:
            return tf.constant(board.who_won(), dtype="float32", shape=(1, 1))
        else:
            return self.model(board.get_board())


    def action(self, board, training, epsilon=0):
        legal_moves = board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        greedy = self.greedy_action(board)
        if training and len(board.history()) >= (2 + (self.player == -1)):
            self.update_model(board, greedy)

        # Exploration
        if random.random() < epsilon:
            move = self.random_action(board)
        else:
            move = greedy

        board.move(*move)

    def update_model(self, board, greedy_move=()):
        if greedy_move == ():
            assert board.who_won() != 2 and board.who_won() != self.player
            self.model.fit(board.history()[-2], self.evaluation(board), batch_size=1, verbose=0)
        else:
            board.move(*greedy_move)
            self.model.fit(board.history()[-3], self.evaluation(board), batch_size=1, verbose=0)
            board.undo_move(*greedy_move)



