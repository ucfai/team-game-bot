from matplotlib import pyplot


def plot_wins(win_states, num, labels=['X', 'O']):
    xs = []
    os = []
    ties = []

    values = [0, 0, 0]

    for i, w in enumerate(win_states):
        if i < num:
            values[w] += 1
        else:
            xs.append(values[1])
            os.append(values[-1])
            ties.append(values[0])

            values[w] += 1
            values[win_states[i-num]] -= 1

    game = range(num, len(xs)+num)
    pyplot.plot(game, xs, label="{} wins".format(labels[0]))
    pyplot.plot(game, os, label="{} wins".format(labels[1]))
    pyplot.plot(game, ties, label="Ties")
    pyplot.legend()
    pyplot.title("Number of Each End State for Previous {} Games".format(num))
    pyplot.show()
