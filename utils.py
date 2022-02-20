from mnk import Board
import datetime

def run_game(agent_train, agent_versing, epsilon=0, training=False, mnk=(3, 3, 3), verbose=False):
    board = Board(*mnk, form="multiplanar-turnflipped", hist_length=-1)
    game = []

    while board.game_ongoing():
        # Select a move
        if board.player == agent_versing.player:
            agent_versing.action(board)
        else:
            agent_train.action(board, training, epsilon)

        # Store game for later analysis
        game.append(board.__str__())

    winner = board.who_won()

    # Back up the terminal state value to the last action chosen by training agent
    if winner != agent_train.player and training:
        agent_train.model.td_update(board, terminal=True)

    if verbose:
        print(board)

    return winner, game

def arg_parser(argv):
    possible_arguments = ["-v", "-mcts"]
    
    # List of booleans representing if each argument is present (in order above)
    present = [1 if arg in argv else 0 for arg in possible_arguments]

    # Last value will be model name
    if len(argv) > 1 and not argv[1].startswith("-"):
        present.append(argv[1])
    else:
        present.append("Model__" + str(datetime.datetime.now())[:-7].replace(" ", "__"))

    return tuple(present)