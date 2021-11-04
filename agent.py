import mnk
import keras.models
import random

class Agent():

    def __init__(self, board, model, player):
        self.board = board;
        self.model = model;
        self.player = player;

    def action(self, epsilon=0.1):
        legal_moves = self.board.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        # Exploration
        if (random.random() < epsilon):
            print("0.1% probability exploration move was made!")
            self.board.move(*legal_moves[random.randint(0, len(legal_moves) - 1)])
            return

        best_move = legal_moves[0]
        max_evaluation = -1

        for move in legal_moves:
            self.board.move(*move)
            evaluation = self.player * self.model(self.board.get_board())
            if evaluation > max_evaluation:
                best_move = move
                max_evaluation = evaluation
            self.board.undo_move(*move)

        self.board.move(*best_move)


