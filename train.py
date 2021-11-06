# TODO: PLOT LOSS CURVES
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import mnk
from agent import Agent
from model import modelXO

games = 1010
m, n, k = 3, 3, 3
epsilon = 1
numEpochs = 1
batchSize = 1
verbose = 0

for game in range(games):
    board = mnk.Board(m, n, k, hist_length=2)

    agentX = Agent(board, modelXO, 1)
    agentO = Agent(board, modelXO, -1)

    move = 1
    while not board.player_has_lost() and len(board.legal_moves()) != 0:
        if move > 2:
            evaluation = modelXO(board.get_board())
            modelXO.fit(board.history()[-2], evaluation, epochs=numEpochs, batch_size=batchSize, verbose=0)

        if board.player == 1:
            agentX.action(epsilon)
        else:
            agentO.action(epsilon)

        if game % 50 == 0:
            print(board)
        move += 1

    terminal_eval = tf.constant(board.who_won(), dtype="float32", shape=(1, 1))
    modelXO.fit(board.history()[-2], terminal_eval, epochs=numEpochs, batch_size=batchSize, verbose=0)
    modelXO.fit(board.history()[-1], terminal_eval, epochs=numEpochs, batch_size=batchSize, verbose=0)

    if game % 300 == 0:
        epsilon /= 10

    if game % 10 == 0:
        print("Game " + str(game) + " goes to " + ['Tie', 'X', 'O'][board.who_won()])

modelXO.save('models/modelXO')
