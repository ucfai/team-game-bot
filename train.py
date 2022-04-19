import numpy as np

from mnk import Board
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from agent import Agent
from model import Model, scheduler
from plot import Diagnostics, save_plots
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

    # Theres a parameter for sample weights. Use if we do importance sampling
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.model.fit(states, target_outputs, verbose=0, callbacks=[lr_scheduler])


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
    agent_train.model.td_update(state, action, board.get_board())

    if verbose:
        print(board)

    return winner, game


def main():
    # Hyperparameter List
    total_games = 100000
    diagnostic_freq = 20
    resample_freq = 10
    hof_gate_freq = 500
    batch_size = 32
    buffer_size = 4000
    epsilon = 0.2  # probability with which a random move is chosen to play

    hof_folder = "menagerie"    # Folder to store the hall-of-fame models
    hof = HOF(mnk, folder=hof_folder)

    print("\nTraining model: {}\n".format(model_name))
    model, diagnostics, games = train(hof, total_games, diagnostic_freq, resample_freq, hof_gate_freq, batch_size, epsilon, buffer_size, Model(mnk))

    save_model(model, model_name)
    save_plots(mnk, hof, model_name, diagnostics)
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
    diagnostics = Diagnostics()
    games = ["" for _ in range(total_games//diagnostic_freq * 2)]

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
            side_best = side_hof * -1

            # Gate the model for HOF
            if game % hof_gate_freq == 0:
                reward, improvement = diagnostics.get_recent_performance()
                if reward > 0 and np.abs(improvement) < 10:
                    hof.gate(model)
                    diagnostics.add_gate_ind()

            if game % diagnostic_freq == 0:
                print("Game: ", game)

                # Run a diagnostic (non-training, no exploration) game to collect data
                diagnostic_winner, game_data = run_diagnostic(model, hof, 1)
                games[game // diagnostic_freq * 2] = game_data
                diagnostics.update_diagnostics(diagnostic_winner, 1)

                diagnostic_winner, game_data = run_diagnostic(model, hof, -1)
                games[game // diagnostic_freq * 2 + 1] = game_data
                diagnostics.update_diagnostics(diagnostic_winner, -1)

    except KeyboardInterrupt:
        print("\n=======================")
        print("Training interrupted.")
        print("=======================")

    print("Training completed.")
    return model, diagnostics, games


def run_diagnostic(model, hof, side_model):
    side_hof = side_model * -1

    model_hof = hof.sample("uniform")
    agent_model = Agent(model, side_model)
    agent_hof = Agent(model_hof, side_hof)

    # Run a diagnostic (non-training, no exploration) game to collect data
    return run_game(agent_model, agent_hof, mnk=mnk, verbose=verbose)


def clear_hof(folder):
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)
        except:
            print("Error while clearing HOF folder.")

if __name__ == "__main__":
    main()
