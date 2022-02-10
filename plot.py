import matplotlib.pyplot as plt

def plot_wins(game_outcomes, run_length, labels=['X', 'O']):
    x_wins = []
    o_wins = []
    ties = []
    num_games = len(game_outcomes)

    values = [0, 0, 0]                                      # Needs better name / documentation

    for game, w in enumerate(game_outcomes):                   # Rename "w"
        if game < run_length:
            values[w] += 1
        else:
            x_wins.append(values[1])
            o_wins.append(values[-1])
            ties.append(values[0])

            values[w] += 1
            values[game_outcomes[game - run_length]] -= 1        # More comments about what run_length is

    game = range(run_length, len(x_wins)+run_length)
    plt.plot(game, x_wins, label="{} wins".format(labels[0]))
    plt.plot(game, o_wins, label="{} wins".format(labels[1]))
    plt.plot(game, ties, label="Ties")
    plt.legend()
    plt.title("Training data for {} Games".format(num_games))
    plt.xlabel("Game number")
    plt.ylabel("Wins out of previous {} games".format(run_length))
