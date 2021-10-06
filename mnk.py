# This class provides a simple interface to work with the rules of m,n,k-games.
# Future games should share a similar interface to make conversion of the AI
# to play different games as seamless as possible.
import numpy as np
from model import model

class Board:
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        self.board = np.zeros((m, n), dtype=int)
        self.empty = 0
        self.player = 1
        self.opponent = -1

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

    def flip_players(self):
        self.player, self.opponent = self.opponent, self.player

    # does a move by changing the board and current player
    def move(self, x, y):
        assert 0 <= x < self.m and 0 <= y < self.n, "Illegal move - Out of bounds"
        assert self.board[x][y] == self.empty, "Illegal move - Spot already taken"
        self.board[x][y] = self.player
        self.flip_players()

    # undoes everything done in the move method
    def undo_move(self, x, y):
        self.board[x][y] = self.empty
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

    # prediction of how good a given board state is for the player
    def evaluate(self):
        return model.predict(self.flatten())

    # plays best ai predicted move
    def play_ai_move(self):
        legal_moves = self.legal_moves()
        assert len(legal_moves) > 0, "No legal moves can be played." 
        best_move = legal_moves[0]
        max_evaluation = 0

        for move in legal_moves:
            self.move(*move)
            evaluation = self.evaluate()[0][0]
            if evaluation > max_evaluation:
                best_move = move
                max_evaluation = evaluation
            self.undo_move(*move)

        self.move(*best_move)

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
