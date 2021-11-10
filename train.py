# TODO: PLOT LOSS CURVES
import tensorflow as tf
import numpy as np
from mnk import Board
import random
from matplotlib import pyplot
from agent import Agent
from model import modelXO
from plot import plot_wins
from hof import HOF


def train(mnk, hof, hof_params, games, diagnostic_freq, epsilon):
    hof.store(modelXO, "init")
    hof_freq, hof_duration = hof_params
    end_states = []
    victories = []

    for game in range(games):
        print(game)
        diagnostic = game % diagnostic_freq == 0

        board = Board(*mnk, form="flatten", hist_length=-1)

        # initialize the agents
        if game % hof_duration == 0:
            modelHOF = hof.sample_hof("limit-uniform")
        sideT = [-1, 1][game % 2]
        sideHOF = [None, -1, 1][sideT]
        agentT = Agent(board, modelXO, sideT, training=not diagnostic)
        agentHOF = Agent(board, modelHOF, sideHOF, training=False)

        while board.who_won() == 2:
            if board.player == sideHOF:
                agentHOF.action()
            else:
                agentT.action(epsilon*(not diagnostic))

        # update value for the last action before the terminal state
        # (only necessary if agent lost, otherwise .action() handles it)
        if board.who_won() != sideT:
            agentT.update_model()

        # occasionally save new model to hall of fame
        if game % hof_freq == 0 and game != 0:
            hof.store(modelXO, game)

        if diagnostic:
            end_states.append(board.who_won())
            victories.append(board.who_won()*sideT)

    return modelXO, end_states, victories


if __name__ == "__main__":
    mnk = (3, 3, 3)
    hof = HOF("menagerie")

    model, end_states, victories = train(mnk, hof, (10, 1), 10000, diagnostic_freq=11, epsilon=0.1)
    model.save('models/modelXO')

    pyplot.subplot(3, 1, 1)
    plot_wins(end_states, 50)

    pyplot.subplot(3, 1, 2)
    plot_wins(victories, 50, ["Best", "HOF"])

    pyplot.subplot(3, 1, 3)
    hof.sample_hist(20)

    pyplot.show()
