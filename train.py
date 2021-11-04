# TODO: PLOT LOSS CURVES
from tensorflow.keras.utils import to_categorical
import numpy as np
import mnk
from agent import Agent
from model import modelXO

games = 150
m, n, k = 3, 3, 3
epsilon = 0.001
numEpochs = 5
batchSize = 1
verbose = 0

for game in range(games):
    board = mnk.Board(m, n, k)

    agentX = Agent(board, modelXO, 1)
    agentO = Agent(board, modelXO, -1)

    while not board.player_has_lost() and len(board.legal_moves()) != 0:

        if board.player == 1:
            agentX.action()
        else:
            agentO.action()

        if game % 5 == 0:
            print(board)

    board_states = np.array(board.history()[:-1])

    if game % 13 == 0:
        verbose = 1
    else:
        verbose = 0

    predicted_next = [modelXO(board_states[i+1]) for i in range(len(board_states)-1)]
    predicted_next.append(np.array([[board.who_won()]]).astype('float32'))
    predicted_next = np.array(predicted_next)

    modelXO.fit(board_states, predicted_next, epochs=numEpochs, batch_size=batchSize, verbose=verbose)
    print("Game " + str(game) + " goes to " + ['Tie','X','O'][board.who_won()])

modelXO.save('models/modelXO')
