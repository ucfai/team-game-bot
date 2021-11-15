# TODO: PLOT LOSS CURVES
from mnk import Board
import random
from matplotlib import pyplot as plt
from agent import Agent
from model import Model
from plot import plot_wins
from hof import HOF

mnk = (3, 3, 3)

# Runs a game from start to end
def run_game(agent_train, agent_versing, epsilon, training):
    board = Board(*mnk, form="flatten", hist_length=-1)
    game = []

    while board.game_ongoing():
        # Select a move
        if board.player == agent_versing.player:
            agent_versing.action(board, False, 0)
        else:
            agent_train.action(board, training, epsilon)
        
        # Store game for later analysis
        game.append(board.__str__())

    winner = board.who_won()

    # Back up the terminal state value to the last action chosen by training agent
    if winner != agent_train.player and training:
        agent_train.model.td_update(board, terminal=True)

    return winner, game


def train(hof, loops, loop_length, epsilon, model):
    end_states = []
    victories = []
    games = []

    # Initialize values
    hof.store(model)
    model_hof = hof.sample()

    # Determine who will play as X and 0
    side_best = [-1, 1][random.random() > 0.5]
    side_hof = side_best * -1

    for loop in range(loops):
        print("\n loop: ",loop)

        # Initialize the agents
        agent_best = Agent(model, side_best)
        agent_hof = Agent(model_hof, side_hof)

        for game in range(loop_length):
            run_game(agent_best, agent_hof, epsilon, training=True)

        # Run a diagnostic (non-training, no exploration) game to collect data
        diagnostic_winner, game_data = run_game(agent_best, agent_hof, 0, training=False)

        # Switch sides for the next loop
        side_best *= -1
        side_hof = side_best * -1

        # Update hall of fame and sample from it for the next loop
        hof.gate(model)
        model_hof = hof.sample("uniform")

        # Store data from loop
        games.append(game_data)
        end_states.append(diagnostic_winner)
        victories.append(diagnostic_winner*side_best)


    return model, end_states, victories, games


if __name__ == "__main__":
    # Initialize hall of fame
    hof = HOF("menagerie")

    num_loops = 10
    loop_length = 5

    # Run training and store final model
    model, end_states, victories, games = train(hof, num_loops, loop_length, 0.2, Model())

    print("Training complete.")
    print("Saving trained model to models/modelXO and chart to plots folder")

    model.save_to('models/modelXO')

    # Create data plots
    plt.subplot(3, 1, 1)
    plot_wins(end_states, 50)

    plt.subplot(3, 1, 2)
    plot_wins(victories, 50, ["Best", "HOF"])

    plt.subplot(3, 1, 3)
    hof.sample_histogram(20)

    plt.show()
    plt.savefig("plots/plot{}.png".format(num_loops * loop_length))

    ind = 0
    while ind != -1:
        ind = int(input("Query a game: "))
        for move in games[ind]:
            print(move)
        pass
