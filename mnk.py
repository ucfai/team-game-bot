# This class provides a simple interface to work with the rules of m,n,k-games.
# Future games should share a similar interface to make conversion of the AI
# to play different games as seamless as possible.
import numpy as np
import random
from model import modelX, modelO
from tensorflow import keras

class Board:
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        self.board = np.zeros((m, n), dtype=int)
        self.empty = 0
        self.player = 1
        self.opponent = -1
        self.board_history = []
        self.modelX = keras.models.load_model('models/modelX')
        self.modelO = keras.models.load_model('models/modelO')
        self.use_saved_models = 1

    # plays best ai predicted move
    def play_ai_move(self, epsilon=0.001):
        legal_moves = self.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played."

        # Exploration
        if (random.random() < epsilon):
            print("0.1% probability exploration move was made!")
            self.move(*legal_moves[random.randint(0, len(legal_moves) - 1)])
            # currently just plays a random legal move
            # ideally have it use monte carlo tree search instead
            return

        best_move = legal_moves[0]
        max_evaluation = 0

        for move in legal_moves:
            self.move(*move)
            evaluation = self.evaluate()
            if evaluation > max_evaluation:
                best_move = move
                max_evaluation = evaluation
            self.undo_move(*move)

        self.move(*best_move)

    # prediction of how good a given board state is for the player
    def evaluate(self):
        if self.player == -1:
            if self.use_saved_models:
                return self.modelX.predict(self.flatten())
            else:
                return modelX.predict(self.flatten())
        else:
            if self.use_saved_models:
                return self.modelO.predict(self.flatten())
            else:
                return modelO.predict(self.flatten())

    def history(self):
        return self.board_history

    def flip_players(self):
        self.player, self.opponent = self.opponent, self.player

    def who_won(self):
        if self.player_has_lost():
            return 'X' if self.player == -1 else 'O'
        elif len(self.legal_moves()) != 0:
            return "Ongoing Game"
        else:
            return "Tie"

    # does a move by changing the board and current player
    def move(self, x, y):
        assert 0 <= x < self.m and 0 <= y < self.n, "Illegal move - Out of bounds"
        assert self.board[x][y] == self.empty, "Illegal move - Spot already taken"
        self.board[x][y] = self.player
        self.board_history.append(self.flatten())
        self.flip_players()

    # undoes everything done in the move method
    def undo_move(self, x, y):
        self.board[x][y] = self.empty
        self.board_history.pop()
        self.flip_players()

    # generates and returns a list of all legal moves
    def legal_moves(self):
        moves = []
        for x, column in enumerate(self.board):
            for y, cell in enumerate(column):
                if cell == self.empty:
                    moves.append((x, y))
        return moves

    # reshapes board into 1-dimensional array for feeding as input to model
    def flatten(self):
        return self.board.reshape(1, self.m * self.n)

    # converting numbers to their respective game values
    @staticmethod
    def print_cast(move):
        return 'O_X'[move + 1]

    # allows for printing of the current board state
    def __str__(self):
        string = ''
        for i, row in enumerate(reversed(list(zip(*self.board)))):
            for x, cell in enumerate(row):
                # avoids printing '_' on bottom edge of grid
                if i == self.n - 1 and cell == self.empty:
                    string += ' '
                else:
                    string += self.print_cast(cell)

                if x != self.m - 1:
                    string += '|'
                else:
                    string += '\n'
        return string

    # returns True if the player whose turn it is has lost, False otherwise
    def player_has_lost(self):
        # check vertical line |
        for column in self.board:
            count = 0
            for cell in column:
                if cell == self.opponent:
                    count += 1
                else:
                    count = 0
                if count == self.k:
                    return True
        # check horizontal line -
        for row in zip(*self.board):
            count = 0
            for cell in row:
                if cell == self.opponent:
                    count += 1
                else:
                    count = 0
                if count == self.k:
                    return True
        # check diagonal line \
        for u in range(self.m + self.n - 1):
            count = 0
            for v in range(u + 1):
                x = u - v
                y = v
                if x >= self.m or y >= self.n:
                    continue
                if self.board[x][y] == self.opponent:
                    count += 1
                else:
                    count = 0
                if count == self.k:
                    return True
        # check diagonal line /
        for u in range(self.m + self.n - 1):
            count = 0
            for v in range(u + 1):
                x = u - v
                y = self.n - 1 - v
                if x >= self.m or y < 0:
                    continue
                if self.board[x][y] == self.opponent:
                    count += 1
                else:
                    count = 0
                if count == self.k:
                    return True
        return False
