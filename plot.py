import matplotlib.pyplot as plt
from model import Model
from agent import Agent
from utils import run_game
import random
import os

def plot_wins(outcomes, model_name, players):

    # We don't plot total wins for each player bc the graph would always increase, making performance evaluation harder.
    # Instead, we plot runs: how many of the previous n games were won. This way, if a model begins performing worse, its line will decrease.

    player1_wins, player2_wins, ties = [], [], []
    run_totals = [0, 0, 0]
    num_games = len(outcomes)
    run_length = max(num_games // 10 , 1)

    for i, outcome in enumerate(outcomes):
        if i < run_length:
            run_totals[outcome] += 1
        else:
            player1_wins.append(run_totals[1])
            player2_wins.append(run_totals[-1])
            ties.append(run_totals[0])

            run_totals[outcome] += 1
            run_totals[outcomes[i - run_length]] -= 1

    game = range(run_length, len(player1_wins) + run_length)

    plt.plot(game, player1_wins, label="{} wins".format(players[0]))
    plt.plot(game, player2_wins, label="{} wins".format(players[1]))
    plt.plot(game, ties, label="Ties")

    plt.legend()
    plt.title("{}: {} diagnostic games".format(model_name, num_games))
    plt.xlabel("Game #")
    plt.ylabel("Wins out of previous {} games".format(run_length))


# 1v1 matrix for historical models: ideally, newer versions beating earlier ones
def winrate_matrix(num_games, step):
    print("Calculating winrate matrix...")
    matrix = []
    for i in range (0, num_games, step):
        matrix.append([])
        for j in range (0, num_games, step):
            model_i = Model("menagerie/{}".format(i))
            model_j = Model("menagerie/{}".format(j))

            side_i = [-1, 1][random.random() > 0.5]
            side_j = side_i * -1

            value = run_game(Agent(model_i, side_i), Agent(model_j, side_j))[0]
            matrix[-1].append(value)

    return matrix


def save_plots(hof, model_name, winnersXO, winnersHOF):

    # Create model's plots folder
    plots_dir = "plots/{}".format(model_name)
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Graph and save each plot
    plt.figure()
    plot_wins(winnersXO, model_name, ['X', 'O'])
    plt.savefig("{}/XO.png".format(plots_dir))
    plt.clf()

    plot_wins(winnersHOF, model_name, ["Best", "HOF"])
    plt.savefig("{}/HOF.png".format(plots_dir))
    plt.clf()

    hof.sample_histogram(20)
    plt.savefig("{}/Sampling.png".format(plots_dir))
    plt.clf()

    num_games = len(winnersXO)
    step = max(1, num_games // 20)
    matrix = winrate_matrix(num_games, step)
    plt.imshow(matrix, cmap="bwr")
    plt.imsave("plots/{}/Matrix.png".format(model_name), matrix, cmap="bwr")
    plt.clf()
