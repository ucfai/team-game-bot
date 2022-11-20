import mnk
import tensorflow as tf
import random

from state_representation import get_input_rep
import output_representation as output_rep


class Agent:

    def __init__(self, model, player):
        self.model = model
        self.player = player    # -1 or 1 representing if playing as X or O

    def greedy_action(self, board):
        legal_moves = board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        action_value_vector = self.model.action_values(get_input_rep(board.get_board()))
        legal_action_values = output_rep.get_legal_vals_obj(board, action_value_vector)
        best_move = max(legal_action_values, key=legal_action_values.get)

        return best_move

    def random_action(self, board):
        legal_moves = board.legal_moves()
        return legal_moves[random.randint(0, len(legal_moves) - 1)]

    def softmax_action(self, board, beta):
        action_value_vector = self.model.action_values(get_input_rep(board.get_board()))
        legal_action_values = output_rep.get_legal_vals_obj(board, action_value_vector)

        legal_val_tensor = tf.constant([list(legal_action_values.values())])
        sampled_ind = tf.random.categorical(tf.math.log(tf.nn.softmax(beta * legal_val_tensor)), 1)[0, 0]

        return list(legal_action_values.keys())[sampled_ind]

    def action(self, board, epsilon=0, beta=None):
        legal_moves = board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        if beta is None:
            best_move = self.greedy_action(board)
        else:
            best_move = self.softmax_action(board, beta)

        # Exploration
        if random.random() < epsilon:
            move = self.random_action(board)
        else:
            move = best_move

        return move

