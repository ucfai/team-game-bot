import tensorflow as tf
import numpy as np


def get_legal_vals(board, q_value_vector):
    move_dict = {}
    q_value_array = np.array(q_value_vector)[0]

    for move in board.legal_moves():
        move_dict[move] = q_value_array[move[0] * board.m + move[1]]

    return move_dict


