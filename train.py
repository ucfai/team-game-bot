# TODO: PLOT LOSS CURVES
from mnk import Board
import random
from matplotlib import pyplot
from agent import Agent
from model import modelXO
from plot import plot_wins
from hof import HOF

mnk = (3, 3, 3)


def run_game(agent_train, agent_verse, epsilon, training):
    board = Board(*mnk, form="flatten", hist_length=-1)

    while board.who_won() == 2:
        if board.player == agent_verse.player:
            agent_verse.action(board, False, 0)
        else:
            agent_train.action(board, training, epsilon)

    winner = board.who_won()

    if winner != agent_train.player and training:
        agent_train.update_model(board)

    return winner


def train(hof, loops, loop_length, epsilon):
    end_states = []
    victories = []

    # initialize values
    hof.store(modelXO, "init")
    model_hof = hof.sample_hof()
    side_best = [-1, 1][random.random() > 0.5]
    side_hof = side_best * -1

    for loop in range(loops):
        print(loop)

        # initialize the agents
        agent_best = Agent(modelXO, side_best)
        agent_hof = Agent(model_hof, side_hof)

        for game in range(loop_length):
            run_game(agent_best, agent_hof, epsilon, training=True)

        diagnostic_winner = run_game(agent_best, agent_hof, 0, training=False)

        if diagnostic_winner != side_hof:
            side_best = [-1, 1][random.random() > 0.5]
            side_hof = side_best * -1
            hof.store(modelXO, loop)
            model_hof = hof.sample_hof()

        end_states.append(diagnostic_winner)
        victories.append(diagnostic_winner)

    return modelXO, end_states, victories


if __name__ == "__main__":
    hof = HOF("menagerie")

    model, end_states, victories = train(hof, 1000, 10, epsilon=0.1)
    model.save('models/modelXO')

    pyplot.subplot(3, 1, 1)
    plot_wins(end_states, 50)

    pyplot.subplot(3, 1, 2)
    plot_wins(victories, 50, ["Best", "HOF"])

    pyplot.subplot(3, 1, 3)
    hof.sample_hist(20)

    pyplot.show()
