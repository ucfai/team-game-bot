# TODO: PLOT LOSS CURVES

import numpy as np
import mnk
from model import modelX, modelO

games = 150
m, n, k = 3, 3, 2
epsilon = 0.001
numEpochs = 5
batchSize = 1
verbose = 0

for game in range(games):
    board = mnk.Board(m, n, k)

    while not board.player_has_lost() and len(board.legal_moves()) != 0:
        board.play_ai_move(epsilon)
        if game % 5 == 0:
            print(board)

    datasetX = np.array(board.history()[::2])
    datasetO = np.array(board.history()[1::2])

    if game % 13 == 0:
        verbose = 1
    else:
        verbose = 0

    if board.who_won() == 'X':
        win = np.ones(len(datasetX))
        loss = np.zeros(len(datasetO))
        modelX.fit(datasetX, win, epochs=numEpochs, batch_size=batchSize, verbose=verbose)
        modelO.fit(datasetO, loss, epochs=numEpochs, batch_size=batchSize, verbose=verbose)
        print("Game " + str(game) + ": X win")
    elif board.who_won() == 'O':
        win = np.ones(len(datasetO))
        loss = np.zeros(len(datasetX))
        modelX.fit(datasetX, loss, epochs=numEpochs, batch_size=batchSize, verbose=verbose)
        modelO.fit(datasetO, win, epochs=numEpochs, batch_size=batchSize, verbose=verbose)
        print("Game " + str(game) + ": O win")
    else:
        tieX = np.zeros(len(datasetX))
        tieO = np.zeros(len(datasetO))
        modelX.fit(datasetX, tieX, epochs=numEpochs, batch_size=batchSize, verbose=verbose)
        modelO.fit(datasetO, tieO, epochs=numEpochs, batch_size=batchSize, verbose=verbose)
        print("Game " + str(game) + ": Tie")

modelX.save('models/modelX')
modelO.save('models/modelO')
