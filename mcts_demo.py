import random
import math
import mnk


def main():
    run_trials(5, 5, 4, 100, 'q', 100)
    run_trials(5, 5, 4, 100, 'n', 100)


def run_trials(m, n, k, iterations, value, trials):
    mcwins = 0
    qlwins = 0
    draws = 0

    for i in range(trials):
        outcome = mc_vs_ql(m, n, k, iterations, value)
        if outcome == 1:
            mcwins += 1
        elif outcome == -1:
            qlwins += 1
        else:
            draws += 1

    print("{} trials for ({}, {}, {}) with {} iterations and using {} values".format(trials, m, n, k, iterations, value))
    print("mcts: ", mcwins)
    print("qlts: ", qlwins)
    print("draw: ", draws)


def mc_vs_ql(m, n, k, iterations, value):
    mc = random.choice([1, -1])

    board = mnk.Board(m, n, k)

    mcroot = Node()
    mcroot.expand(board.legal_moves())

    qlroot = Node()
    qlroot.expand(board.legal_moves())

    while board.who_won() == 2:
        if board.player == mc:
            move = mctsAI(board, mcroot, iterations, value).last_move

            qlroot = qlroot.move(move)
            mcroot = mcroot.move(move)

            board.move(*move)
        else:
            move = qltsAI(board, qlroot, iterations, value).last_move

            qlroot = qlroot.move(move)
            mcroot = mcroot.move(move)

            board.move(*move)

    winner = board.who_won()
    return winner * mc

def play_human():
    m, n, k = map(int, input('Choose m, n, and k: ').split())
    iterations = int(input('Choose AI Iterations: '))
    human = random.choice([1, -1])

    board = mnk.Board(m, n, k)

    root = Node()
    root.expand(board.legal_moves())

    print(board)
    while board.who_won() == 2:
        if board.player == human:
            x, y = map(int, input('Make a move (x y): ').split())
            x -= 1
            y -= 1
            root = root.move((x, y))
            board.move(x, y)
        else:
            print('AI is thinking')
            root = qltsAI(board, root, iterations, 'n')
            board.move(*root.last_move)
        print(board)
        winner = board.who_won()

        if winner == human:
            print('You win!')
        elif winner == -human:
            print('You lose :(')
        elif winner == 0:
            print('You tie.')


def mctsAI(board, node, iterations, value):
    for i in range(iterations):
        mcts(board, node)

    if value == 'q':
        return max(node.children, key=lambda child: child.q)
    else:
        return max(node.children, key=lambda child: child.n)


def qltsAI(board, node, iterations, value):
    for i in range(iterations):
        qlts(board, node)

    if value == 'q':
        return max(node.children, key=lambda child: child.q)
    else:
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

    node.n += 1
    node.q = node.q + (1 / node.n) * (winner * -board.player - node.q)

    return winner


def qlts(board, node):
    if node.isLeaf:
        winner = rollout(board)
        if board.who_won() == 2:
            node.expand(board.legal_moves())
        node.n += 1
        node.q += winner * -board.player
    else:
        next_state = node.max_child()
        board.move(*next_state.last_move)
        qlts(board, next_state)
        board.undo_move()

        target = -max(node.children, key=lambda child: child.q).q
        node.n += 1
        node.q = node.q + (1 / node.n) * (target - node.q)


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
        self.q = 0
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
        return self.q + 2 * math.sqrt(math.log(N + 1) / (self.n + 1))
    
    def max_child(self):
        return max(self.children, key=lambda child: child.UCT(self.n))


if __name__ == '__main__':
    main()
