# This class provides a simple interface to work with the rules of m,n,k-games.
# Future games should share a similar interface to make conversion of the AI
# to play different games as seamless as possible.
import numpy as np


class Board:
    def __init__(self, m, n, k, hist_length=-1, state=None):
        if state is None:
            self.board = np.zeros((m, n), dtype=int)
            self.player, self.opponent = 1, -1
        else:
            self.board, self.player = state
            self.opponent = self.player * -1

        self.m = m
        self.n = n
        self.k = k
        self.hist_length = hist_length
        self.empty = 0
        self.board_history = []
        self.undo_buffer = np.zeros((m, n), dtype=int)
        self.move_history = []

    def shape(self):
        return self.m, self.n

    def history(self):
        return self.board_history

    def add_history(self):
        if self.hist_length == -1 or len(self.board_history) < self.hist_length:
            self.board_history.append(self.get_board())
        else:
            self.undo_buffer = self.board_history[0]
            for i in range(len(self.board_history)-1):
                self.board_history[i] = self.board_history[i+1]
            self.board_history[-1] = self.get_board()

    def del_history(self):
        if self.hist_length == -1 or len(self.board_history) < self.hist_length:
            self.board_history.pop()
        else:
            for i in range(0, len(self.board_history)-1, -1):
                self.board_history[i+1] = self.board_history[i]
            self.board_history[0] = self.undo_buffer
            self.undo_buffer = np.zeros((self.m, self.n), dtype=int)

    def flip_players(self):
        self.player, self.opponent = self.opponent, self.player

    def num_legal_moves(self):
            return len(self.legal_moves())

    def who_won(self):
        if self.player_has_lost():
            return 1 if self.player == -1 else -1
        elif len(self.legal_moves()) != 0:
            # ongoing
            return 2
        else:
            # draw
            return 0

    # Does a move by changing the board and current player
    def move(self, x, y):
        assert 0 <= x < self.m and 0 <= y < self.n, "Illegal move - Out of bounds"
        assert self.board[x][y] == self.empty, "Illegal move - Spot already taken"
        self.board[x][y] = self.player
        self.add_history()
        self.flip_players()
        self.move_history.append((x, y))

    # Undoes everything done in the move method
    def undo_move(self):
        x, y = self.move_history.pop()
        self.board[x][y] = self.empty
        self.del_history()
        self.flip_players()

    # Generates and returns a list of all legal moves
    def legal_moves(self):
        moves = []
        for x, column in enumerate(self.board):
            for y, cell in enumerate(column):
                if cell == self.empty:
                    moves.append((x, y))
        return moves

    def num_legal_moves(self):
        return len(self.legal_moves())

    # Reshapes board into the form needed for the model
    def get_board(self):
        return self.board, self.player

    def game_ongoing(self):
        return not (self.player_has_lost() or (self.num_legal_moves() == 0))

    # Converting numbers to their respective game values
    @staticmethod
    def print_cast(move):
        return 'O_X'[move + 1]

    # Allows for printing of the current board state
    def __str__(self):
        string = ''
        for i, row in enumerate(reversed(list(zip(*self.board)))):
            for x, cell in enumerate(row):
                string += self.print_cast(cell)
                if x != self.m - 1:
                    string += '|'
                else:
                    string += '\n'
        return string

    def game_ongoing(self):
        return not ( self.player_has_lost() or (self.num_legal_moves() == 0) )

    # returns True if the player whose turn it is has lost, False otherwise
    def player_has_lost(self):
        if len(self.move_history) == 0:
            return False
        last_x, last_y = self.move_history[-1]
        # check vertical line |
        count = 0
        for y in range(self.n):
            if self.board[last_x][y] == self.opponent:
                count += 1
            else:
                count = 0
            if count == self.k:
                return True
        # check horizontal line -
        count = 0
        for x in range(self.m):
            if self.board[x][last_y] == self.opponent:
                count += 1
            else:
                count = 0
            if count == self.k:
                return True
        # check diagonal line \
        x_plus_y = last_x + last_y
        count = 0
        for x in range(self.m):
            y = x_plus_y - x
            if y < 0 or y >= self.n:
                continue
            if self.board[x][y] == self.opponent:
                count += 1
            else:
                count = 0
            if count == self.k:
                return True
        # check diagonal line /
        x_minus_y = last_x - last_y
        count = 0
        for x in range(self.m):
            y = x - x_minus_y
            if y < 0 or y >= self.n:
                continue
            if self.board[x][y] == self.opponent:
                count += 1
            else:
                count = 0
            if count == self.k:
                return True
        return False
