import numpy as np

from mnk import Board
import random
import matplotlib.pyplot as plt
from agent import Agent
from model import Model
from plot import plot_wins, save_plots
from hof import HOF
from replay_buffer import ReplayBuffer
from state_representation import get_input_rep
from utils import run_game, arg_parser
from save_model import save_model
import sys
import os
import shutil

# Set cmd-line training arguments
verbose, mcts, model_name = arg_parser(sys.argv)
model_name = "new_model"
mnk = (3, 3, 3)


def train_on_replays(model, batch):
    states = []
    target_outputs = []
    for experience in batch:
        target_outputs.append(model.get_target(*experience))
        states.append(get_input_rep(experience[0])[0])

    states = np.asarray(states)

    target_outputs = np.asarray(target_outputs)

    # Theres a parameter for train_on_batch for sample weights. Use if we do importance sampling
    model.model.fit(states, target_outputs, verbose=0)


def run_training_game(agent_train, agent_versing, replay_buffer, epsilon=0, mnk=(3, 3, 3), verbose=False):
    board = Board(*mnk, hist_length=-1)
    game = []
    state, action = None, None

    while board.game_ongoing():
        # Select a move
        if board.player == agent_versing.player:
            board.move(*agent_versing.action(board))
        else:
            move = agent_train.action(board, epsilon)

            if state is not None and action is not None:
                agent_train.model.td_update(state, action, board.get_board())
                replay_buffer.store((state, action, board.get_board()))
                train_on_replays(agent_train.model, replay_buffer.sample())

            state, action = board.get_board(), move
            board.move(*move)

        # Store game for later analysis
        game.append(board.__str__())

    winner = board.who_won()

    # Back up the terminal state value to the last action chosen by training agent
    if winner != agent_train.player:
        agent_train.model.td_update(state, action, board.get_board())

    if verbose:
        print(board)

    return winner, game


def main():
    # Hyperparameter List
    total_games = 50000
    diagnostic_freq = 20
    resample_freq = 10
    hof_gate_freq = 2000
    batch_size = 32
    buffer_size = 4000
    epsilon = 0.2  # probability with which a random move is chosen to play

    hof_folder = "menagerie"    # Folder to store the hall-of-fame models
    hof = HOF(mnk, folder=hof_folder)

    print("\nTraining model: {}\n".format(model_name))
    model, winnersXO, winnersHOF, games = train(hof, total_games, diagnostic_freq, resample_freq, hof_gate_freq, batch_size, epsilon, buffer_size, Model(mnk))

    save_model(model, model_name)
    save_plots(mnk, hof, model_name, winnersXO, winnersHOF)
    clear_hof(hof_folder)

    # Can be used after looking at plot to analyze important milestones
    ind = 0                                                                          # Put into a function
    while ind != -1:
        ind = int(input("Query a game: "))

        if ind >= len(games):
            print("Too large. Try again")
            continue

        for move in games[ind]:
            print(move)
        pass


def train(hof, total_games, diagnostic_freq, resample_freq, hof_gate_freq, batch_size, epsilon, buffer_size, model):
    winnersXO = [0 for _ in range(total_games//diagnostic_freq)]
    winnersHOF = [0 for _ in range(total_games//diagnostic_freq)]
    games = ["" for _ in range(total_games//diagnostic_freq)]

    # Initialize hall of fame
    hof.store(model)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size, batch_size)

    try:
        for game in range(total_games):
            if game % resample_freq == 0:
                side_best = [-1, 1][random.random() > 0.5]
                side_hof = side_best * -1
                model_hof = hof.sample("uniform")

            # Initialize the agents
            agent_best = Agent(model, side_best)
            agent_hof = Agent(model_hof, side_hof)

            # Play game and train on its outcome
            run_training_game(agent_best, agent_hof, replay_buffer, epsilon, mnk)

            # Switch sides for next game
            side_hof *= -1
            side_best *= -1

            # Gate the model for HOF
            if game % hof_gate_freq == 0:
                hof.gate(model)

            if game % diagnostic_freq == 0:
                print("Game: ", game)

                # Resample hof so diagnostic is not biased towards last game played
                temp_side_best = [-1, 1][random.random() > 0.5]
                temp_side_hof = side_best * -1

                temp_model_hof = hof.sample("uniform")
                temp_agent_best = Agent(model, temp_side_best)
                temp_agent_hof = Agent(temp_model_hof, temp_side_hof)

                # Run a diagnostic (non-training, no exploration) game to collect data
                diagnostic_winner, game_data = run_game(temp_agent_best, temp_agent_hof, mnk=mnk, verbose=verbose)

                # Store data from diagnostic game for this batch
                games[game//diagnostic_freq] = game_data
                winnersXO[game//diagnostic_freq] = diagnostic_winner              # X or O
                winnersHOF[game//diagnostic_freq] = diagnostic_winner*side_best   # Best or HOF

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
