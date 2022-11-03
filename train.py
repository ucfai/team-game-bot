import numpy as np

from mnk import Board
import random
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from agent import Agent
from model import Model, scheduler
from plot import Diagnostics, save_plots
from hof import HOF
from replay_buffer import ReplayBuffer
from state_representation import get_input_rep
import output_representation as output_rep
from utils import run_game, arg_parser, save_model
import sys
import os
import shutil
import warnings

# Set cmd-line training arguments
verbose, mcts, model_name = arg_parser(sys.argv)
model_name = "new_model"
mnk = (3, 3, 3)


def get_corrected_action_values(model, lagging_model, state, action, next_state, reward, terminal):
    """Returns an action value vector with a 1 step TD update as a target for training

    Args:
        model: Model object being trained
        state: Board representing the previous state of the game.
        action: Move played after previous state.
        next_state: Next state of the game after action was taken.
    """

    m, n, k = model.mnk

    # TODO: Is this actually necessary (converting state to a Board object)? Might be wasteful
    start_board = Board(*model.mnk, state=state)
    next_board = Board(*model.mnk, state=next_state)

    prev_output = model.action_values(start_board)

    # OPT 1: If this line is used, illegal actions will be ignored.
    target_output = np.copy(prev_output)

    # OPT 2: If this is used, illegal actions will be trained to have action value -1.
    # ________________________________________________________________________________
    # target_output = np.full(shape=prev_output.shape, fill_value=-1, dtype='float32')
    #
    # for move in start_board.legal_moves():
    #    index = move[0] * m + move[1]
    #    target_output[0][index] = prev_output[0][index]

    # If next board is terminal, evaluate the real value of the state. Otherwise, use a
    # Double Q-learning update (argmax using the slow model, obtain the value of the
    # resulting action using the current model. This prevents over-estimation bias)
    if terminal:
        td_target = tf.constant(reward, dtype="float32", shape=(1, 1))
    else:
        legal_slow_action_values = output_rep.get_legal_vals(next_board, lagging_model.action_values(next_board))
        argmax_move = max(legal_slow_action_values, key=legal_slow_action_values.get)
        td_target = model.action_values(next_board)[0][argmax_move[0] * next_board.n + argmax_move[1]]

    target_output[0][action[0] * n + action[1]] = td_target
    return target_output


def train_on_replays(model, lagging_model, batch):
    """Trains the model with 1 step TD updates on a batch of samples.

    Args:
        model: Model object being trained
        batch: Batch of (state, action, next_state) tuples being trained on
    """
    states = []
    target_outputs = []

    # Experiences are tuples (state, action, state')
    for experience in batch:
        target_outputs.append(get_corrected_action_values(model, lagging_model, *experience))
        states.append(get_input_rep(experience[0])[0])

    states = np.asarray(states)
    target_outputs = np.asarray(target_outputs)

    # Theres a parameter for sample weights. Use if we do importance sampling
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model.model.fit(states, target_outputs, epochs=1, batch_size=len(states), steps_per_epoch=1, callbacks=[lr_scheduler], verbose=False)


def run_training_game(transitions, agent_train, agent_versing, lagging_model, replay_buffer, n_steps=1, model_update_freq=4, lagging_freq=100, start_at=5000, epsilon=0, mnk=(3, 3, 3), verbose=False):
    """Runs a training game with the provided agents.

    Args:
        agent_train: Agent object being trained
        agent_versing: Agent object being played against (taken from HOF)
        replay_buffer: Replay buffer object used to store moves and obtain training batches
        epsilon: Chance of the training agent performing a random action
        mnk: Board parameters
        verbose: Whether to print the final board
    """

    board = Board(*mnk, hist_length=-1)
    game = []

    # State queue used for multi-step targets
    state_queue = []

    while board.game_ongoing():
        # Select a move
        if board.player == agent_versing.player:
            board.move(*agent_versing.action(board))
        else:
            transitions += 1
            move = agent_train.action(board, epsilon)

            if len(state_queue) >= n_steps:
                # Adds last action to replay buffer
                replay_buffer.store((*state_queue[0], board.get_board(), 0, False))

            if transitions % model_update_freq == 0 and transitions > start_at:
                # Trains on a replay batch
                train_on_replays(agent_train.model, lagging_model, replay_buffer.sample())
            if transitions % lagging_freq == 0:
                # Updates the lagging model to the current model
                lagging_model.model = tf.keras.models.clone_model(agent_train.model.model)

            state_queue.append((board.get_board(), move))
            if len(state_queue) > n_steps:
                state_queue.pop(0)

            board.move(*move)

        # Store game for later analysis
        if verbose:
            game.append(board.__str__())

    winner = board.who_won()

    # Back up the terminal state value to the last actions chosen by training agent
    while len(state_queue) > 0:
        reward = agent_train.player * winner
        if reward == 0:
            reward = 0

        replay_buffer.store((*state_queue.pop(0), board.get_board(), reward, True))

    if verbose:
        print(board)

    return winner, game, transitions


def train(hof, total_games, diagnostic_freq, resample_freq, hof_gate_freq, batch_size, epsilon_i, epsilon_f, decay_period, buffer_size, n_steps, update_freq, lagging_freq, start_transition, model, lr):
    diagnostics = Diagnostics()
    games = ["" for _ in range(total_games // diagnostic_freq * 2)]
    epsilon_step = (epsilon_f - epsilon_i) / decay_period
    epsilon = epsilon_i

    # Initialize hall of fame
    hof.store(model)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size, batch_size)

    # Initialize lagging model
    lagging_model = Model(mnk, model=tf.keras.models.clone_model(model.model))
    transitions = 0

    try:
        for game in range(total_games):
            epsilon += epsilon_step
            # Regularly choose a new HOF opponent
            if game % resample_freq == 0:
                side_best = [-1, 1][random.random() > 0.5]
                side_hof = side_best * -1
                model_hof = hof.sample("uniform")

            # Initialize the agents
            agent_best = Agent(model, side_best)
            agent_hof = Agent(model_hof, side_hof)

            # Play game and train on its outcome
            _, _, transitions = run_training_game(transitions, agent_best, agent_hof, lagging_model, replay_buffer, n_steps, update_freq, lagging_freq, start_transition, epsilon, mnk)

            # Switch sides for next game
            side_hof *= -1
            side_best = side_hof * -1

            # Regularly attempt to add the model into HOF ("gating")
            if game % hof_gate_freq == 0:
                reward, improvement = diagnostics.get_recent_performance()

                # Only add if reward is positive and improvement has plateaued
                if reward > 0 and np.abs(improvement) < 0.025:
                    epsilon = epsilon_i
                    K.set_value(model.model.opt.learning_rate, lr)
                    hof.store(model)

                    # Adds red line for when new models are added in plots
                    diagnostics.add_gate_ind()

            if game % diagnostic_freq == 0:
                print("Game: ", game)

                # Run diagnostic (non-training, no exploration) games to collect data
                # One game is played as player 1, one as player 2, for every HOF model
                avg_win = avg_x = avg_o = avg_t = avg_hof = 0

                for i in range(hof.pop_size):
                    model_hof = hof.sample(index=i)

                    diagnostic_winner, game_data = run_diagnostic(model, model_hof, 1)
                    games[game // diagnostic_freq * 2] = game_data

                    avg_win += diagnostic_winner
                    if diagnostic_winner == 1:
                        avg_x += 1
                        avg_t += 1
                    elif diagnostic_winner == -1:
                        avg_o += 1
                        avg_hof += 1

                    diagnostic_winner, game_data = run_diagnostic(model, model_hof, -1)
                    games[game // diagnostic_freq * 2 + 1] = game_data

                    avg_win += -diagnostic_winner
                    if diagnostic_winner == 1:
                        avg_x += 1
                        avg_hof += 1
                    elif diagnostic_winner == -1:
                        avg_o += 1
                        avg_t += 1

                diagnostics.update_reward(avg_win / (hof.pop_size * 2))
                diagnostics.update_xo(avg_x / (hof.pop_size * 2), avg_o / (hof.pop_size * 2))
                diagnostics.update_outcome(avg_t / (hof.pop_size * 2), avg_hof / (hof.pop_size * 2))

    except KeyboardInterrupt:
        print("\n=======================")
        print("Training interrupted.")
        print("=======================")

    print("Training completed.")
    return model, diagnostics, games


def run_diagnostic(model, model_hof, side_model):
    """Runs a diagnostic game with the provided models (no model training). Used to collect data
    on model performance.

    Args:
        model: Model being trained
        model_hof: HOF model to oppose
        side_model: The side the training model is playing as (1 or -1)
    """
    side_hof = side_model * -1
    agent_model = Agent(model, side_model)
    agent_hof = Agent(model_hof, side_hof)

    return run_game(agent_model, agent_hof, mnk=mnk, verbose=verbose)


# Deletes entries in HOF folder
def clear_hof(folder):
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)
        except:
            print("Error while clearing HOF folder (Specified folder not found).")


def main():
    # Hyperparameter List
    diagnostic_freq = 25  # How often to run diagnostic games (in games)
    resample_freq = 100  # How often to choose a new HOF opponent (in games)
    hof_gate_freq = 2000  # How often to gate a new model into the HOF (in games)

    total_games = 100000  # Total num of training games
    batch_size = 32  # Batch size for training
    lr = 0.01  # Learning rate for SGD

    update_freq = 4  # How often to train the model on a replay batch (in moves)
    buffer_size = 50000  # Num of moves to store in replay buffer
    n_steps = 1  # Num of steps used for temporal difference training targets
    lagging_freq = 500  # How often to update the lagging model (in moves)
    start_transition = 5000

    epsilon_i = 0.1  # Probability with which a random move is chosen to play
    epsilon_f = 0.01
    decay_period = 10000

    hof_folder = "menagerie"    # Folder to store the hall-of-fame models
    hof = HOF(mnk, folder=hof_folder)

    print("\nTraining model: {}\n".format(model_name))
    model, diagnostics, games = train(hof, total_games, diagnostic_freq, resample_freq, hof_gate_freq, batch_size, epsilon_i, epsilon_f, decay_period, buffer_size, n_steps, update_freq, lagging_freq, start_transition, Model(mnk, lr=lr), lr=lr)

    save_model(model, model_name)
    save_plots(mnk, hof, model_name, diagnostics)
    clear_hof(hof_folder)

    # Can be used after looking at plot to analyze important milestones
    # TODO: Put into a function
    ind = 0
    while ind != -1:
        ind = int(input("Query a game: "))

        if ind >= len(games):
            print("Too large. Try again")
            continue

        for move in games[ind]:
            print(move)
        pass


if __name__ == "__main__":
    main()
