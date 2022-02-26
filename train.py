from mnk import Board
import random
import matplotlib.pyplot as plt
from agent import Agent
from model import Model
from plot import plot_wins, save_plots
from hof import HOF
from utils import run_game, arg_parser
from save_model import save_model
import sys
import os
import shutil

# Set cmd-line training arguments
verbose, mcts, model_name = arg_parser(sys.argv)
mnk = (3, 3, 3)

def main():

    # Hyperparameter List
    num_batches = 20_000        # Total training games = num_batches * games_per_batch
    games_per_batch = 5
    epsilon = 0.2               # Epsilon is the exploration factor: probability with which a random move is chosen to play

    hof_folder = "menagerie"    # Folder to store the hall-of-fame models
    hof = HOF(mnk, folder=hof_folder)

    print("\nTraining model: {}\n".format(model_name))
    model, winnersXO, winnersHOF, games = train(hof, num_batches, games_per_batch, epsilon, Model())

    save_model(model, model_name)
    save_plots(hof, model_name, winnersXO, winnersHOF)
    clear_hof(hof_folder)

    # Can be used after looking at plot to analyze important milestones
    ind = 0                                                                          # Put into a function
    while ind != -1:
        ind = int(input("Query a game: "))
        for move in games[ind]:
            print(move)
        pass


def train(hof, num_batches, games_per_batch, epsilon, model):
    winnersXO = []
    winnersHOF = []
    games = []

    # Initialize hall of fame
    hof.store(model)

    try:
        for batch_number in range(num_batches):
            print("Batch:", batch_number, "(Games {}-{})".format(batch_number * games_per_batch + 1, (batch_number + 1) * games_per_batch))

            # Runs a batch of games, after which we can play/save a diagnostic game to see if it improved and store current model to hof
            for game in range(games_per_batch):

                # Randomly assign sides (X or O) for game to be played
                side_best = [-1, 1][random.random() > 0.5]
                side_hof = side_best * -1

                model_hof = hof.sample("uniform")

                # Initialize the agents
                agent_best = Agent(model, side_best)
                agent_hof = Agent(model_hof, side_hof)

                # Play game and train on its outcome
                run_game(agent_best, agent_hof, epsilon, training=True)

            # Gate will determine if model is worthy, and store in hof only if it is (Currently, it just stores every game)
            hof.gate(model)

            # Switch sides and resample hof so diagnostic is not biased towards last game played
            side_best *= -1
            side_hof = side_best * -1
            model_hof = hof.sample("uniform")
            agent_best = Agent(model, side_best)
            agent_hof = Agent(model_hof, side_hof)

            # Run a diagnostic (non-training, no exploration) game to collect data
            diagnostic_winner, game_data = run_game(agent_best, agent_hof, 0, training=False, mnk=mnk, verbose=verbose)

            # Store data from diagnostic game for this batch
            games.append(game_data)
            winnersXO.append(diagnostic_winner)            # X or O
            winnersHOF.append(diagnostic_winner*side_best)   # Best or HOF

    except KeyboardInterrupt:
        print("\n=======================")
        print("Training interrupted.")
        print("=======================")

    print("Training completed.")
    return model, winnersXO, winnersHOF, games

def clear_hof(folder):
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)
        except:
            print("Error while clearing HOF folder.")

if __name__ == "__main__":
    main()
