# This class provides a simple interface to work with the rules of m,n,k-games.
# Future games should share a similar interface to make conversion of the AI
# to play different games as seamless as possible.
class Board:
    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        self.board = [['_']*n for _ in range(m)]
        self.player = 'X'
        self.opponent = 'O'
    def flip_players(self):
        self.player, self.opponent = self.opponent, self.player
    # does a move by changing the board and current player
    def move(self, x, y):
        self.board[x][y] = self.player
        self.flip_players()
    # undoes everything done in the move method
    def undo_move(self, x, y):
        self.board[x][y] = '_'
        self.flip_players()
    # generates a list of all legal moves
    def generate_moves(self):
        moves = []
        for x, column in enumerate(self.board):
            for y, cell in enumerate(column):
                if cell == '_':
                    moves.append((x, y))
        return moves
    # allows for printing of the current board state
    def __repr__(self):
        string = ''
        for row in reversed(list(zip(*self.board))):
            for x, cell in enumerate(row):
                string += cell
                if x != self.m - 1:
                    string += '|'
                else:
                    string += '\n'
        return string
    # returns True if the player whose turn it is has lost, False otherwise
    def lost(self):
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
        for row in list(zip(*self.board)):
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
