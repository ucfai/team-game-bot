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

        best_move = legal_moves[0]
        max_evaluation = -1

        for move in legal_moves:
            val = self.model.action_value(board, move)
            if val > max_evaluation:
                best_move = move
                max_evaluation = val

        return best_move

    def random_action(self, board):
        legal_moves = board.legal_moves()
        return legal_moves[random.randint(0, len(legal_moves) - 1)]

    def action(self, board, training=False, epsilon=0):
        legal_moves = board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        greedy_move = self.greedy_action(board)
        if training:
            self.model.td_update(board, greedy_move)

        # Exploration
        if random.random() < epsilon:
            move = self.random_action(board)
        else:
            move = greedy_move

        board.move(*move)

