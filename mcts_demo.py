import random
import math
import mnk

def main():
    m, n = map(int, input('Size? (width height): ').split())
    k = int(input('#-in-a-row?: '))
    iterations = int(input('AI iterations?: '))
    human = random.choice([1, -1])
    player = 1

    board = mnk.Board(m, n, k)
    root = Node()
    print(board)
    while True:
        if root.isLeaf:
                root.expand(board.legal_moves())
        if board.player == human:
            x, y = map(int, input('Move? (x y): ').split())
            x -= 1
            y -= 1
            root = root.move((x, y))
            board.move(x, y)
        else:
            print('AI moves')
            root = AI(board, root, iterations)
            # print(board.legal_moves())
            board.move(*root.last_move)
        print(board)
        winner = board.who_won()
        if winner == human:
            print('You win!')
            break
        elif winner == -human:
            print('You lose :(')
            break
        elif winner == 0:
            print('You tie.')
            break

def AI(board, node, iterations):
    for i in range(iterations):
       mcts(board, node)
    max_n = -1
    for child in node.children:
        # print('Move: {0}, w: {1}, n: {2}'.format(child.last_move, child.n - child.w, child.n))
        if child.n > max_n:
            max_n = child.n
            max_child = child
    return max_child

def mcts(board, node):
    if node.isLeaf:
        moves_played = []
        while True:
            winner = board.who_won()
            if winner != 2:
                break
            legal_moves = board.legal_moves()
            if len(legal_moves) == 0:
                winner = 0
                break
            move = random.choice(legal_moves)
            board.move(*move)
            moves_played.append(move)
        for move in reversed(moves_played):
            board.undo_move(*move)
        if board.who_won() == 2:
            node.expand(board.legal_moves())
    else:
        max_UCT = -1
        for child in node.children:
            UCT = child.UCT(node.n, math.sqrt(2))
            if UCT > max_UCT:
                max_UCT = UCT
                max_child = child
        board.move(*max_child.last_move)
        winner = mcts(board, max_child)
        board.undo_move(*max_child.last_move)
    if winner == board.player:
        node.w += 1
    if winner == 0:
        node.w += 0.5
    node.n += 1
    return winner

class Node:
    def __init__(self, last_move = None):
        self.last_move = last_move
        self.w = 0
        self.n = 0
        self.children = []
        self.isLeaf = True

    def move(self, move):
        for child in self.children:
            if child.last_move == move:
                return child

    def expand(self, legal_moves):
        for move in legal_moves:
            self.children.append(Node(move))
            self.isLeaf = False

    def UCT(self, N, c):
        return (self.n - self.w) / (self.n + 1) + c * math.sqrt(math.log(N + 1) / (self.n + 1))

if __name__ == '__main__':
    main()
