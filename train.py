# TODO: PLOT LOSS CURVES
from mnk import Board
import random
from matplotlib import pyplot
from agent import Agent
from model import modelXO
from plot import plot_wins
from hof import HOF

mnk = (15, 15, 5)


def run_game(agent_train, agent_verse, epsilon, training):
    board = Board(*mnk, form="multiplanar", hist_length=-1)
    game = []

    while board.who_won() == 2:
        if board.player == agent_verse.player:
            agent_verse.action(board, False, 0)
        else:
            agent_train.action(board, training, epsilon)
        
        game.append(board.__str__())

    winner = board.who_won()

    if winner != agent_train.player and training:
        agent_train.update_model(board)

    return winner, game


def train(hof, loops, loop_length, epsilon):
    base_epsilon = epsilon
    end_states = []
    victories = []
    games = []

    # initialize values
    hof.store(modelXO, "init")
    model_hof = hof.sample_hof()
    # side_best = [-1, 1][random.random() > 0.5]
    side_best = -1
    side_hof = side_best * -1

    loops_stuck = 0

    for loop in range(loops):
        print("\n loop: ",loop)

        # initialize the agents
        agent_best = Agent(modelXO, side_best)
        agent_hof = Agent(model_hof, side_hof)

        print("__ running diagnostic __")
        diagnostic_winner, game_data = run_game(agent_best, agent_hof, 0, training=False)
        print("diagnostic winner: {}, our model: {}".format(diagnostic_winner,side_best))

        if diagnostic_winner != side_best:
            loops_stuck += 1

            for game in range(loop_length):
                run_game(agent_best, agent_hof, epsilon, training=True)

            print("epsilon: ", epsilon)
            epsilon = 0.6 + (epsilon-0.6)/1.1
        else:
            print("********** diagnostic passed. resampling **********")

            side_best = [-1, 1][random.random() > 0.5]
            side_hof = side_best * -1
            if loops_stuck > 0:
                hof.store(modelXO, loop)
            model_hof = hof.sample_hof("limit-uniform")

            epsilon = base_epsilon
            loops_stuck = 0

        games.append(game_data)
        end_states.append(diagnostic_winner)
        victories.append(diagnostic_winner*side_best)


    return modelXO, end_states, victories, games


if __name__ == "__main__":
    hof = HOF("menagerie")

    model, end_states, victories, games = train(hof, 100, 5, epsilon=0.01)
    model.save('models/modelXO')

    pyplot.subplot(3, 1, 1)
    plot_wins(end_states, 50)

    pyplot.subplot(3, 1, 2)
    plot_wins(victories, 50, ["Best", "HOF"])

    pyplot.subplot(3, 1, 3)
    hof.sample_hist(20)

    pyplot.show()

    ind = 0
    while ind != -1:
        ind = int(input("Query a game"))
        for move in games[ind]:
            print(move)
        pass
