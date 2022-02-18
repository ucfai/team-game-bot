import random
import math

def main():
    width, height = map(int, input('Size? (width height): ').split())
    order = int(input('#-in-a-row?: '))
    iterations = int(input('AI iterations?: '))
    human = random.choice('XO')
    player = 'X'

    grid = [['_']*height for x in range(width)]
    board = Board(grid, width, height, order)
    root = Node(board, player)
    root.print()
    while True:
        if root.player == human:
            print()
            x, y = map(int, input('Move? (x y): ').split())
            x -= 1
            y -= 1
            if root.isLeaf:
                root.expand()
            root = root.move(x, y)
        else:
            root = AI(root, iterations)
        if root.isLeaf:
            root.expand()
        root.print()
        winner = root.winner()
        if winner == human:
            print()
            print('You win!')
            break
        elif winner == flip(human):
            print()
            print('You lose :(')
            break
        elif len(root.children) == 0:
            print()
            print('You tie.')
            break

def flip(player):
    if player == 'X':
        return 'O'
    if player == 'O':
        return 'X'

class Board:
    def __init__(self, grid, width, height, order):
        self.grid = grid
        self.width = width
        self.height = height
        self.order = order
    def copy(self):
        grid_copy = [row.copy() for row in self.grid]
        return Board(grid_copy, self.width, self.height, self.order)
    def __getitem__(self, x):
        return self.grid[x]
    def print(self):
        print()
        for y in range(self.height-1, -1, -1):
            for x in range(self.width):
                print(self[x][y], end='')
                if x != self.width-1:
                    print('|', end='')
                else:
                    print()

def mcts(node):
    if node.isLeaf:
        winner = node.rollout() 
        if node.winner() == 'incomplete':
            node.expand()
    else:
        max_UCT = -1
        for child in node.children:
            UCT = child.UCT(node.n, math.sqrt(2))
            if UCT > max_UCT:
                max_UCT = UCT
                max_child = child
        winner = mcts(max_child)
    if winner == node.player:
        node.w += 1
    if winner == 'draw':
        node.w += 0.5
    node.n += 1
    return winner

def AI(node, iterations):
    for i in range(iterations):
       mcts(node)
    max_n = -1
    for child in node.children:
        # child.print()
        # print(child.n - child.w, child.n)
        if child.n > max_n:
            max_n = child.n
            max_child = child
    return max_child

class Node:
    def __init__(self, board, player):
        self.board = board
        self.player = player
        self.isLeaf = True
        self.w = 0
        self.n = 0
        self.children = []
        self.solved = False
    def print(self):
        self.board.print()
    def UCT(self, N, c):
        return (self.n - self.w) / (self.n + 1) + c * math.sqrt(math.log(N + 1) / (self.n + 1))
    def expand(self):
        for y in range(self.board.height):
            for x in range(self.board.width):
                if self.board[x][y] == '_':
                    board_copy = self.board.copy()
                    board_copy[x][y] = self.player
                    self.children.append(Node(board_copy, flip(self.player)))
                    self.isLeaf = False
        if self.isLeaf:
            self.solved = True
    def move(self, x, y):
        for node in self.children:
            if node.board[x][y] == self.player:
                return node
    def rollout(self):
        board_copy = self.board.copy()
        current_player = self.player
        while True:
            winner = Node.board_winner(board_copy)
            if winner != 'incomplete':
                return winner
            possible_moves = []
            for y in range(board_copy.height):
                for x in range(board_copy.width):
                    if self.board[x][y] == '_':
                        possible_moves.append((x, y))
            if len(possible_moves) == 0:
                return 'draw'
            x, y = random.choice(possible_moves)
            board_copy[x][y] = current_player
            current_player = flip(current_player)
    def winner(self):
        return Node.board_winner(self.board)
    def board_winner(board):
        for player in 'XO':
            # horizontal -
            for y in range(board.height):
                count = 0
                for x in range(board.width):
                    if board[x][y] == player:
                        count += 1
                    else:
                        count = 0
                    if count == board.order:
                        return player
            # vertical |
            for x in range(board.width):
                count = 0
                for y in range(board.height):
                    if board[x][y] == player:
                        count += 1
                    else:
                        count = 0
                    if count == board.order:
                        return player
            # diagonal \
            for x_plus_y in range(board.width + board.height - 1):
                count = 0
                for x in range(board.width):
                    y = x_plus_y - x
                    if y < 0 or y >= board.height:
                        continue
                    if board[x][y] == player:
                        count += 1
                    else:
                        count = 0
                    if count == board.order:
                        return player
            # diagonal /
            for x_minus_y in range(1 - board.height, board.width):
                count = 0
                for x in range(board.width):
                    y = x - x_minus_y
                    if y < 0 or y >= board.height:
                        continue
                    if board[x][y] == player:
                        count += 1
                    else:
                        count = 0
                    if count == board.order:
                        return player
        # incomplete
        for x in range(board.width):
            for y in range(board.height):
                if board[x][y] == '_':
                    return 'incomplete'
        # draw
        return 'draw'

if __name__ == '__main__':
    main()
