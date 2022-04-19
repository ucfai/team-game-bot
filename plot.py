import matplotlib.pyplot as plt
from model import Model
from agent import Agent
from utils import run_game
import random
import os


class Diagnostics:
    def __init__(self, run_length=100):
        self.run_length = run_length
        self.xo_outcomes = []
        self.model_outcomes = []
        self.reward_totals = []
        self.reward_deltas = []
        self.gating_indices = []
        self.index = 0

    def update_diagnostics(self, outcome, player):
        self.xo_outcomes.append(outcome)

        reward = outcome*player
        self.model_outcomes.append(reward)

        self.reward_totals.append(reward)
        self.reward_deltas.append(reward)

        if self.index > 0:
            self.reward_totals[-1] += self.reward_totals[-2]
            self.reward_deltas[-1] += self.reward_deltas[-2]

        if self.index >= self.run_length:
            self.reward_totals[-1] -= self.model_outcomes[self.index - self.run_length]
            self.reward_deltas[-1] -= 2 * self.model_outcomes[self.index - self.run_length]

        if self.index >= 2 * self.run_length:
            self.reward_deltas[-1] += self.model_outcomes[self.index - 2 * self.run_length]

        self.index += 1

    def add_gate_ind(self):
        self.gating_indices.append(self.index)

    def get_recent_performance(self):
        if self.index == 0:
            return 0, 0

        return self.reward_totals[-1], self.reward_deltas[-1]


def plot_wins(outcomes, model_name, players):
    # We don't plot total wins for each player bc the graph would always increase, making performance evaluation harder.
    # Instead, we plot runs: how many of the previous n games were won. This way, if a model begins performing worse, its line will decrease.

    player1_wins, player2_wins, ties = [], [], []
    run_totals = [0, 0, 0]
    num_games = len(outcomes)
    run_length = 100

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


# Vertical lines where the model was gated
def add_gating_markers(gating_indices):
    for ind in gating_indices:
        plt.axvline(x=ind, c='red')


# Displays a histogram of the model iterations sampled from the hall of fame
def sample_histogram(sample_history, bins=100):
    plt.hist(sample_history, bins)
    plt.title("Sampling of Model Indices from HOF")
    plt.show()


# 1v1 matrix for historical models: ideally, newer versions beating earlier ones
def winrate_matrix(mnk, num_games, step):
    print("Calculating winrate matrix... (may take a few mins)")
    matrix = []
    for i in range (0, num_games, step):
        matrix.append([])
        for j in range (0, num_games, step):
            model_i = Model(mnk, "menagerie/{}".format(i))
            model_j = Model(mnk, "menagerie/{}".format(j))

            side_i = [-1, 1][random.random() > 0.5]
            side_j = side_i * -1

            value = run_game(Agent(model_i, side_i), Agent(model_j, side_j))[0]
            matrix[-1].append(value)

    return matrix


def save_plots(mnk, hof, model_name, diagnostics):

    # Create model's plots folder
    plots_dir = "plots/{}".format(model_name)
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)

    # Graph and save each plot
    plt.figure()
    plot_wins(diagnostics.xo_outcomes, model_name, ['X', 'O'])
    add_gating_markers(diagnostics.gating_indices)
    plt.savefig("{}/XO.png".format(plots_dir))
    plt.clf()

    plot_wins(diagnostics.model_outcomes, model_name, ["Best", "HOF"])
    add_gating_markers(diagnostics.gating_indices)
    plt.savefig("{}/HOF.png".format(plots_dir))
    plt.clf()

    plt.plot(range(diagnostics.index), diagnostics.reward_totals)
    add_gating_markers(diagnostics.gating_indices)
    plt.title("{}: Reward for {} diagnostic games".format(model_name, diagnostics.index+1))
    plt.xlabel("Game #")
    plt.ylabel("Cumulative reward over previous {} games".format(diagnostics.run_length))
    plt.savefig("{}/Reward.png".format(plots_dir))
    plt.clf()

    plt.plot(range(diagnostics.index), diagnostics.reward_deltas)
    add_gating_markers(diagnostics.gating_indices)
    plt.title("{}: Cumulative reward derivative for {} diagnostic games".format(model_name, diagnostics.index+1))
    plt.xlabel("Game #")
    plt.ylabel("Difference in cumulative reward for previous two {} length runs".format(diagnostics.run_length))
    plt.savefig("{}/Improvement.png".format(plots_dir))
    plt.clf()

    sample_histogram(hof.sample_history, 20)
    plt.savefig("{}/Sampling.png".format(plots_dir))
    plt.clf()

    num_games = diagnostics.index
    step = max(1, hof.pop_size // 40)
    matrix = winrate_matrix(mnk, hof.pop_size, step)
    plt.imshow(matrix, cmap="bwr")
    plt.imsave("plots/{}/Matrix.png".format(model_name), matrix, cmap="bwr")
    plt.clf()
