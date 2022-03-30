import mnk
import keras.models
import tensorflow as tf
import random

import output_representation as output_rep


class Agent:

    def __init__(self, model, player):
        self.model = model
        self.player = player    # -1 or 1 representing if playing as X or O

    def greedy_action(self, board):
        legal_moves = board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        action_value_vector = self.model.action_values(board)
        legal_action_values = output_rep.get_legal_vals(board, action_value_vector)
        best_move = max(legal_action_values, key=legal_action_values.get)

        return best_move

    def random_action(self, board):
        legal_moves = board.legal_moves()
        return legal_moves[random.randint(0, len(legal_moves) - 1)]

    def action(self, board, epsilon=0):
        legal_moves = board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        greedy_move = self.greedy_action(board)

        # Exploration
        if random.random() < epsilon:
            move = self.random_action(board)
        else:
            move = greedy_move

        return move

