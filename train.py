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
from enum import Enum

# Set cmd-line training arguments
verbose, mcts, model_name = arg_parser(sys.argv)
mnk = (3, 3, 3)
plot_folder = "plots/{}".format(model_name)
hof_folder = "menagerie/{}".format(model_name)    # Folder to store the hall-of-fame models

class ResetType(Enum):
    NONE = 0  # Reset nothing
    OPT = 1  # Reset only optimizer
    FULL = 2  # Reset model and optimizer

class HyperParameters():
    def __init__(self, diagnostic_rate, training_sample_rate, diagnostic_run_length, hof_resample_rate, hof_gate_rate, hof_wait_time, total_games, batch_size, 
        lr, buffer_size, alpha, buffer_beta, min_priority, update_rate, n_steps, lagging_period, training_wait_time, epsilon, policy_beta, reset_type, plotting_rate):

        # Diagnostic params
        self.diagnostic_rate = diagnostic_rate  # How often diagnostic games are run (in # of games)
        self.diagnostic_run_length = diagnostic_run_length  # Run length over which diagnostic returns are averaged
        self.training_sample_rate = training_sample_rate  # How often to sample a training reward for diagnostics
        self.plotting_rate = plotting_rate

        self.hof_resample_rate = hof_resample_rate  # How often to sample a new HOF opponent (in # of games)
        self.hof_gate_rate = hof_gate_rate  # How often to check if new model can be added to HOF (in # of games)
        self.hof_wait_time = hof_wait_time  # Minimum time between HOF additions (in # of games)
        
        self.total_games = total_games  # Total number of training games
        self.batch_size = batch_size  # Batch size (taken from replay buffer)
        self.lr = lr  # Learning rate for optimizer

        self.buffer_size = buffer_size  # Max number of transitions to store in replay buffer
        self.alpha = alpha  # Weight that high priority transitions are given when sampling from buffer (alpha > 0)
        self.buffer_beta = buffer_beta  # Degree of importance sampling used to counteract off-policy sampling (0 <= beta <= 1
        self.min_priority = min_priority  # Minimum priority given to a transition

        self.update_rate = update_rate  # How often to perform a training update on the model (in # of moves)
        self.n_steps = n_steps  # Num of steps used in temporal difference bootstrapping
        self.lagging_period = lagging_period  # How often to update the lagging model (in # of moves)
        self.training_wait_time = training_wait_time  # How many moves to wait before continuing training after HOF additions
        self.reset_type = reset_type  # What kind of reset to perform on model after HOF addition

        self.epsilon = epsilon  # Chance of picking a random move during training
        self.policy_beta = policy_beta  # The lower this is, the closer the policy is to random. The higher, the closer it is to greedy (policy_beta > 0)

def get_corrected_action_values(model, states, actions, td_errors, weights):
    """Returns an action value vector with a 1 step TD update as a target for training

    Args:
        model: Model object being trained
        state: Board representing the previous state of the game.
        action: Move played after previous state.
        next_state: Next state of the game after action was taken.
    """

    m, n, k = model.mnk

    prev_outputs = model.action_values(states)

    # Illegal actions will be ignored. This could be changed to assign -1 to illegal actions
    # but would likely hinder training.
    target_outputs = np.copy(prev_outputs)

    for i in range(target_outputs.shape[0]):
        target_outputs[i][actions[i]] += weights[i] * td_errors[i]

    return target_outputs


def train_on_replays(model, lagging_model, replay_buffer, params):
    """Trains the model with 1 step TD updates on a batch of samples.

    Args:
        model: Model object being trained
        batch: Batch of (state, action, next_state) tuples being trained on
    """

    m, n, k = model.mnk
    batch_size = replay_buffer.batch_size
    batch, importance_sampling = replay_buffer.sample_batch()

    states = np.zeros(shape=(batch_size, m, n, 2))
    next_states = np.zeros(shape=(batch_size, m, n, 2))
    actions = np.zeros(batch_size, dtype="int32")
    rewards = np.zeros(batch_size, dtype="float32")
    terminal = np.zeros(batch_size, dtype="?")

    # Experiences are tuples (state, action, state')
    for i, experience in enumerate(batch):
        states[i], actions[i], next_states[i], rewards[i], terminal[i] = experience

    bootstrap_vals = np.zeros(batch_size, dtype="float32")

    state_action_vals = model.action_values(states)
    state_action_vals = np.array([state_action_vals[i][action] for i, action in enumerate(actions)])

    next_state_action_vals = lagging_model.action_values(next_states)
    _, argmax_inds = model.state_value(next_states, terminal)

    for i in range(batch_size):
        bootstrap_vals[i] = 0 if argmax_inds[i] == -1 else next_state_action_vals[i][argmax_inds[i]]

    td_errors = bootstrap_vals + rewards - state_action_vals
    weights = tf.math.pow(importance_sampling, params.buffer_beta)

    priorities = tf.math.abs(td_errors) + tf.constant(params.min_priority, dtype=tf.float32, shape=(batch_size))
    priorities = tf.math.pow(priorities, params.alpha)
    replay_buffer.update_batch(priorities)

    target_outputs = get_corrected_action_values(model, states, actions, td_errors, weights)

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    model.model.fit(states, target_outputs, epochs=1, batch_size=len(states), steps_per_epoch=1, callbacks=[lr_scheduler], verbose=False)

def run_training_game(diagnostics, transitions, agent_train, agent_versing, lagging_model, replay_buffer, params, mnk=(3, 3, 3), verbose=False, plot_reward=False):
    """Runs a training game with the provided agents.

    Args:
        agent_train: Agent object being trained
        agent_versing: Agent object being played against (taken from HOF)
        replay_buffer: Replay buffer object used to store moves and obtain training batches
        epsilon: Chance of the training agent performing a random action
        mnk: Board parameters
        verbose: Whether to print the final board
    """
    m, n, k = mnk
    board = Board(m, n, k, hist_length=-1)
    game = []

    # State queue used for multi-step targets
    state_queue = []

    while board.game_ongoing():
        # Select a move
        if board.player == agent_versing.player:
            board.move(*agent_versing.greedy_action(board))
        else:
            transitions += 1
            move = agent_train.action(board, params.epsilon, params.policy_beta)

            if len(state_queue) >= params.n_steps:
                # Adds last action to replay buffer
                state, action = state_queue[0]
                replay_buffer.store((get_input_rep(state)[0], action[0] * n + action[1], get_input_rep(board.get_board())[0], 0, False))

            if transitions % params.update_rate == 0 and transitions > params.training_wait_time:
                # Trains on a replay batch
                train_on_replays(agent_train.model, lagging_model, replay_buffer, params)
            if transitions % params.lagging_period == 0:
                # Updates the lagging model to the current model
                lagging_model.model = tf.keras.models.clone_model(agent_train.model.model)

            state_queue.append((board.get_board(), move))
            if len(state_queue) > params.n_steps:
                state_queue.pop(0)

            board.move(*move)

        # Store game for later analysis
        if verbose:
            game.append(board.__str__())

    winner = board.who_won()
    reward = agent_train.player * winner

    if plot_reward:
        diagnostics.update_training(reward)

    # Back up the terminal state value to the last actions chosen by training agent
    while len(state_queue) > 0:
        state, action = state_queue.pop(0) 
        replay_buffer.store((get_input_rep(state)[0], action[0] * n + action[1], get_input_rep(board.get_board())[0], reward, True))

    return winner, game, transitions


def train(hof, params, model):
    diagnostics = Diagnostics(run_length=params.diagnostic_run_length)
    games = ["" for _ in range(params.total_games // params.diagnostic_rate * 2)]

    # Initialize hall of fame
    hof.store(model)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(params.buffer_size, params.batch_size, params.alpha)

    # Initialize lagging model
    lagging_model = Model(mnk, model=tf.keras.models.clone_model(model.model))

    transitions = 0
    best_diagnostic = -1
    games_since_hof = params.hof_wait_time

    try:
        for game in range(params.total_games):
            games_since_hof += 1

            # Regularly choose a new HOF opponent
            if game % params.hof_resample_rate == 0:
                side_best = [-1, 1][random.random() > 0.5]
                side_hof = side_best * -1
                model_hof = hof.sample(index=game % hof.pop_size)

            # Initialize the agents
            agent_best = Agent(model, side_best)
            agent_hof = Agent(model_hof, side_hof)

            # Play game and train on its outcome
            plot_reward = (game % params.training_sample_rate == 0)
            _, _, transitions = run_training_game(diagnostics, transitions, agent_best, agent_hof, lagging_model, replay_buffer, params, mnk, plot_reward=plot_reward)

            # Switch sides for next game
            side_hof *= -1
            side_best = side_hof * -1

            assert side_hof != side_best, "Opponents can't be on the same side"

            # Regularly attempt to add the model into HOF ("gating")
            if game % params.hof_gate_rate == 0 and games_since_hof > params.hof_wait_time:
                reward, improvement = diagnostics.get_recent_performance()

                # Only add if reward is positive and improvement has plateaued
                if (reward > 0 and reward == best_diagnostic and np.abs(improvement) == 0) or reward == 1:
                    print("\nAdding model to HOF...")
                    hof.store(model)
                    # Adds red line for when new models are added in plots
                    diagnostics.add_gate_ind()

                    replay_buffer.clear()
                    transitions = 0
                    games_since_hof = 0
                    best_diagnostic = -1

                    if params.reset_type is ResetType.OPT:
                        model.reset_optimizer()
                    elif params.reset_type is ResetType.FULL:
                        model.initialize_model()

                    print("Done.\n")

            if game % params.diagnostic_rate == 0:
                print("Game: ", game)

                # Run diagnostic (non-training, no exploration) games to collect data
                # One game is played as player 1, one as player 2, for every HOF model
                avg_win = avg_x = avg_o = avg_t = avg_hof = 0

                for i in range(hof.pop_size):
                    model_hof = hof.sample(index=i)

                    diagnostic_winner, game_data = run_diagnostic(model, model_hof, 1)
                    # games[game // diagnostic_freq * 2] = game_data

                    avg_win += diagnostic_winner
                    if diagnostic_winner == 1:
                        avg_x += 1
                        avg_t += 1
                    elif diagnostic_winner == -1:
                        avg_o += 1
                        avg_hof += 1

                    diagnostic_winner, game_data = run_diagnostic(model, model_hof, -1)
                    # games[game // diagnostic_freq * 2 + 1] = game_data

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

                best_diagnostic = max(best_diagnostic, diagnostics.rewards[-1])
                print("Real Reward: {}, Smoothed Reward: {}, Improvement: {}".format(diagnostics.rewards[-1], *diagnostics.get_recent_performance()))

            if game % params.plotting_rate == 0:
                save_model(model, model_name)
                save_plots(mnk, hof, plot_folder, hof_folder, model_name, diagnostics)


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

    return run_game(agent_model, agent_hof, mnk=mnk, verbose=False)


# Deletes entries in HOF folder
def clear_hof(folder):
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)
        except:
            print("Error while clearing HOF folder (Specified folder not found).")



def main():
    # Hyperparameter List
    diagnostic_rate = 10  # How often to run diagnostic games (in number of games)
    diagnostic_run_length = 40 # Run length for diagnostic smoothing (in diagnostic games)
    training_sample_rate = 7
    plotting_rate = 10000

    hof_resample_rate = 100  # How often to choose a new HOF opponent (in games)
    hof_gate_rate = 1000  # How often to gate a new model into the HOF (in games)
    hof_wait_time = 4000 # How long to wait after adding to HOF before adding again

    total_games = 200000  # Total num of training games
    batch_size = 32  # Batch size for training lr = 0.001  # Learning rate for SGD
    lr = 0.001

    buffer_size = 20000  # Num of moves to store in replay buffer
    alpha = 0.7
    buffer_beta = 0.5
    min_priority = 0.01

    update_rate = 4  # How often to train the model on a replay batch (in moves)
    n_steps = 1  # Num of steps used for temporal difference training targets
    lagging_period = 500  # How often to update the lagging model (in moves)
    training_wait_time = 5000
    reset_type = ResetType.OPT

    epsilon = 0.0  # Chance of picking a random move
    policy_beta = 1.0  # The lower this is, the more likely a "worse" move is chosen (don't set < 0)

    params = HyperParameters(diagnostic_rate=diagnostic_rate, training_sample_rate=training_sample_rate, diagnostic_run_length=diagnostic_run_length, hof_resample_rate=hof_resample_rate,
            hof_gate_rate=hof_gate_rate, hof_wait_time=hof_wait_time, total_games=total_games, batch_size=batch_size, lr=lr, buffer_size=buffer_size, alpha=alpha, buffer_beta=buffer_beta, 
            min_priority=min_priority, update_rate=update_rate, n_steps=n_steps, lagging_period=lagging_period, training_wait_time=training_wait_time, epsilon=epsilon, policy_beta=policy_beta,
            reset_type=reset_type, plotting_rate=plotting_rate)

    hof_folder = "menagerie/{}".format(model_name)    # Folder to store the hall-of-fame models
    plot_folder = "plots/{}".format(model_name)

    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    with open("{}/hyperparams.txt".format(plot_folder), "w") as text_file:
            print(f"{params.__dict__}", file=text_file)

    clear_hof(hof_folder)
    hof = HOF(mnk, folder=hof_folder)

    print("\nTraining model: {}\n".format(model_name))
    model, diagnostics, games = train(hof, params, Model(mnk, lr=params.lr))

    save_model(model, model_name)
    save_plots(mnk, hof, plot_folder, hof_folder, model_name, diagnostics)

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
