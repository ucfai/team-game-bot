import random
import math
import mnk

def main():
    m, n, k = map(int, input('Choose m, n, and k: ').split())
    iterations = int(input('Choose AI Iterations: '))
    human = random.choice([1, -1])

    board = mnk.Board(m, n, k)
    root = Node()
    print(board)
    while True:
        if root.isLeaf:
                root.expand(board.legal_moves())
        if board.player == human:
            x, y = map(int, input('Make a move (x y): ').split())
            x -= 1
            y -= 1
            root = root.move((x, y))
            board.move(x, y)
        else:
            print('AI is thinking')
            root = AI(board, root, iterations)
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
    for _ in range(iterations):
       mcts(board, node)
    return max(node.children, key=lambda child: child.n)

def mcts(board, node):
    if node.isLeaf:
        winner = rollout(board)
        if board.who_won() == 2:
            node.expand(board.legal_moves())
    else:
        next_state = node.max_child()
        board.move(*next_state.last_move)
        winner = mcts(board, next_state)
        board.undo_move()
    if winner == board.player:
        node.w += 1
    if winner == 0:
        node.w += 0.5
    node.n += 1
    return winner

def rollout(board):
    moves_played = 0
    while True:
        winner = board.who_won()
        if winner != 2:
            break
        legal_moves = board.legal_moves()
        move = random.choice(legal_moves)
        board.move(*move)
        moves_played += 1
    for _ in range(moves_played):
        board.undo_move()
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

    def UCT(self, N):
        return (self.n - self.w) / (self.n + 1) + math.sqrt(2 * math.log(N + 1) / (self.n + 1))
    
    def max_child(self):
        return max(self.children, key=lambda child: child.UCT(self.n))

if __name__ == '__main__':
    main()
