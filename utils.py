from mnk import Board
import datetime


def save_model(model, model_name):
    print("Saving trained model to models/{}".format(model_name))
    model.save_to('models/{}'.format(model_name))


def run_game(agent_train, agent_versing, mnk=(3, 3, 3), verbose=False):
    board = Board(*mnk, hist_length=-1)
    game = []

    while board.game_ongoing():
        # Select a move
        if board.player == agent_versing.player:
            board.move(*agent_versing.action(board))
        else:
            board.move(*agent_train.action(board))

        # Store game for later analysis
        game.append(board.__str__())

    if verbose:
        print(board)

    return board.who_won(), game


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