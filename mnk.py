# This class provides a simple interface to work with the rules of m,n,k-games.
# Future games should share a similar interface to make conversion of the AI
# to play different games as seamless as possible.
import numpy as np
import random

class Board:
    def __init__(self, m, n, k, flatten = True, hist_length = -1):
        self.m = m
        self.n = n
        self.k = k
        self.flatten = flatten
        self.hist_length = hist_length
        self.board = np.zeros((m, n), dtype=int)
        self.empty = 0
        self.player = 1
        self.opponent = -1
        self.board_history = []
        self.undo_buffer = np.zeros((m, n), dtype=int)

    def history(self):
        return self.board_history

    def add_history(self):
        if self.hist_length == -1 or len(self.board_history) < self.hist_length:
            self.board_history.append(self.get_board())
        else:
            self.undo_buffer = self.board_history[0]
            for i in range(len(self.board_history)-1):
                self.board_history[i] = board_history[i+1]
            self.board_history[-1] = self.get_board()

    def del_history(self):
        if self.hist_length == -1 or len(self.board_history) < self.hist_length:
            self.board_history.pop()
        else:
            for i in range(0,len(self.board_history)-1,-1):
                self.board_history[i+1] = self.board_history[i]
            self.board_history[0] = self.undo_buffer
            self.undo_buffer = np.zeros((m, n), dtype=int)


    def flip_players(self):
        self.player, self.opponent = self.opponent, self.player

    def who_won(self):
        if self.player_has_lost():
            return 1 if self.player == -1 else -1
        elif len(self.legal_moves()) != 0:
            return "Ongoing Game"
        else:
            return 0

    # does a move by changing the board and current player
    def move(self, x, y):
        assert 0 <= x < self.m and 0 <= y < self.n, "Illegal move - Out of bounds"
        assert self.board[x][y] == self.empty, "Illegal move - Spot already taken"
        self.board[x][y] = self.player
        self.add_history()
        self.flip_players()

    # undoes everything done in the move method
    def undo_move(self, x, y):
        self.board[x][y] = self.empty
        self.del_history()
        self.flip_players()

    # generates and returns a list of all legal moves
    def legal_moves(self):
        moves = []
        for x, column in enumerate(self.board):
            for y, cell in enumerate(column):
                if cell == self.empty:
                    moves.append((x, y))
        return moves

    # reshapes board into 1-dimensional array for feeding as input to model if flatten is True
    def get_board(self):
        if self.flatten:
            return self.board.reshape(1, self.m * self.n)
        else:
            return self.board

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
