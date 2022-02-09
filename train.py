# TODO: PLOT LOSS CURVES
from mnk import Board
import random
import matplotlib.pyplot as plt
from agent import Agent
from model import Model
from plot import plot_wins
from hof import HOF
from utils import run_game

mnk = (3, 3, 3)


def main():
    # Initialize hall of fame
    hof = HOF(mnk, "menagerie")

    num_loops = 20000
    loop_length = 5

    # Run training and store final model
    model, end_states, victories, games = train(hof, num_loops, loop_length, 0.2, Model())

    print("Training complete.")
    print("Saving trained model to models/modelXO and chart to plots folder")

    model.save_to('models/modelXO')

    # Create data plots
    plt.figure()
    plt.subplot(3, 1, 1)
    plot_wins(end_states, 100)

    plt.subplot(3, 1, 2)
    plot_wins(victories, 100, ["Best", "HOF"])

    plt.subplot(3, 1, 3)
    hof.sample_histogram(20)
    plt.savefig("plots/plot{}.png".format(num_loops * loop_length))

    print("Calculating winrate matrix")
    hof.winrate_matrix(150)
    plt.show()

    ind = 0
    while ind != -1:
        ind = int(input("Query a game: "))
        for move in games[ind]:
            print(move)
        pass


def train(hof, loops, loop_length, epsilon, model):
    end_states = []
    victories = []
    games = []

    # Initialize values
    hof.store(model)
    model_hof = hof.sample()

    try:
        for loop in range(loops):
            print("\n loop: ", loop)

            side_best = [-1, 1][random.random() > 0.5]
            side_hof = side_best * -1

            for game in range(loop_length):
                # Initialize the agents
                agent_best = Agent(model, side_best)
                agent_hof = Agent(model_hof, side_hof)

                run_game(agent_best, agent_hof, epsilon, training=True)

                # Switch sides for the next game
                side_best = [-1, 1][random.random() > 0.5]
                side_hof = side_best * -1

                model_hof = hof.sample("uniform")

            # Update hall of fame and sample from it for the next loop
            hof.gate(model)

            side_best *= -1
            side_hof = side_best * -1

            agent_best = Agent(model, side_best)
            agent_hof = Agent(model_hof, side_hof)

            # Run a diagnostic (non-training, no exploration) game to collect data
            diagnostic_winner, game_data = run_game(agent_best, agent_hof, 0, training=False, mnk=mnk)

            # Store data from loop
            games.append(game_data)
            end_states.append(diagnostic_winner)
            victories.append(diagnostic_winner*side_best)
    except KeyboardInterrupt:
        print("Training interrupted")

    return model, end_states, victories, games


if __name__ == "__main__":
    main()
