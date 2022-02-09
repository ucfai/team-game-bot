from mnk import Board


def run_game(agent_train, agent_versing, epsilon=0, training=False, mnk=(3, 3, 3)):
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

    return winner, game
