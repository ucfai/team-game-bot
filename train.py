# TODO: PLOT LOSS CURVES
import tensorflow as tf
import numpy as np
import mnk
import random
from agent import Agent
from model import modelXO
from plot import plot_wins
from hof import HOF


m, n, k = 3, 3, 3
hof = HOF("menagerie")
hof.store(modelXO, "init")
modelHOF = hof.sample_hof()

hof_freq = 10  # how often to save the model to the HOF
hof_duration = 2  # how long to keep using the same HOF model before loading a new one

games = 1000
epsilon = 0.1  # exploration constant
decay_freq = 10  # how often to decrease epsilon
decay_factor = 0.00099  # how much to decrease by

end_states = []
victories = []
stored_games = []

for game in range(games):
    board = mnk.Board(m, n, k, flatten=False, hist_length=-1)

    # decrease exploration over time
    if game % decay_freq == 0 and game != 0:
        epsilon -= decay_factor

    # initialize the agents
    if game % hof_duration == 0 and game != 0:
        modelHOF = hof.sample_hof()
    sideT = [-1, 1][random.random() > 0.5]
    sideHOF = [None, -1, 1][sideT]
    agentT = Agent(board, modelXO, sideT)
    agentHOF = Agent(board, modelHOF, sideHOF)

    move = 1
    while not board.player_has_lost() and len(board.legal_moves()) != 0:
        # have the appropriate agent select a move
        if board.player == sideHOF:
            agentHOF.action(epsilon)
        else:
            agentT.action(epsilon)

        # back up the current board evaluation to the last action chosen by the current agent
        if move > 2:
            evaluation = modelXO(board.get_board())
            modelXO.fit(board.history()[-3], evaluation, batch_size=1, verbose=0)
        move += 1

        if game % 50 == 0:
            print(board)

    # back up the terminal state value to the last actions chosen by either agent
    terminal_eval = tf.constant(board.who_won(), dtype="float32", shape=(1, 1))
    modelXO.fit(board.history()[-3], terminal_eval, batch_size=1, verbose=0)
    modelXO.fit(board.history()[-2], terminal_eval, batch_size=1, verbose=0)

    # occasionally save new model to hall of fame
    if game % hof_freq == 0 and game != 0:
        hof.store(modelXO, game)

    end_states.append(board.who_won())
    victories.append(board.who_won()*sideT)
    if game % 10 == 0:
        print("Game {} goes to {} ({})".format(str(game), ["tie", "best", "hof"][board.who_won()*sideT], ['Tie', 'X', 'O'][board.who_won()]))

plot_wins(end_states, 50)
plot_wins(victories, 50, ["Best", "HOF"])
modelXO.save('models/modelXO')
