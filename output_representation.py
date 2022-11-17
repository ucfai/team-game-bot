import tensorflow as tf
import numpy as np


def get_legal_vals_obj(board, q_value_vector):
    move_dict = {}
    q_value_array = np.array(q_value_vector)[0]

    for move in board.legal_moves():
        move_dict[move] = q_value_array[move[0] * board.n + move[1]]

    return move_dict

def legal_argmax(input_state, q_value_vector):
    summed = np.sum(input_state, axis=2)
    m, n = summed.shape

    has_legal = False
    max_ind = 0
    max_val = q_value_vector[max_ind]

    for i in range(q_value_vector.shape[0]):
        if summed[i // n][i % n] == 0 and q_value_vector[i] > max_val:
            has_legal = True
            max_ind = i
            max_val = q_value_vector[max_ind]

    if not has_legal:
        max_ind = -1

    return max_val, max_ind
    


