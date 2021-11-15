from matplotlib import pyplot as plt


def plot_wins(game_outcomes, run_length, labels=['X', 'O']):
    xs = []
    os = []
    ties = []

    values = [0, 0, 0]

    for i, w in enumerate(game_outcomes):
        if i < run_length:
            values[w] += 1
        else:
            xs.append(values[1])
            os.append(values[-1])
            ties.append(values[0])

            values[w] += 1
            values[game_outcomes[i-run_length]] -= 1

    game = range(run_length, len(xs)+run_length)
    plt.plot(game, xs, label="{} wins".format(labels[0]))
    plt.plot(game, os, label="{} wins".format(labels[1]))
    plt.plot(game, ties, label="Ties")
    plt.legend()
    plt.title("Number of Each End State for Previous {} Games".format(run_length))
    plt.xlabel("Game number")
    plt.ylabel("Wins out of previous {} games".format(run_length))


