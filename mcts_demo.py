import random
import math
import mnk


def main():
    run_trials(MCTree, QLTree, 5, 5, 4, 100, 'q', (2, 2), 1000)


def run_trials(tree1Gen, tree2Gen, m, n, k, iterations, value, exploration, trials):
    t1wins = 0
    t2wins = 0
    draws = 0

    for i in range(trials):
        tree1 = tree1Gen()
        tree2 = tree2Gen()
        outcome = compare_trees(tree1, tree2, m, n, k, iterations, value, exploration)
        if outcome == 1:
            t1wins += 1
        elif outcome == -1:
            t2wins += 1
        else:
            draws += 1

        print("{} trials complete".format(i+1))

    print("{} trials for ({}, {}, {}) with {} iterations, using {} values and exploration {}".format(trials, m, n, k, iterations, value, exploration))
    print("tree 1: ", t1wins/trials * 100)
    print("tree 2: ", t2wins/trials * 100)
    print("draw: ", draws/trials * 100)


def compare_trees(tree1, tree2, m, n, k, iterations, value, c):
    c1, c2 = c
    t1 = random.choice([1, -1])

    board = mnk.Board(m, n, k)

    tree1.expand(board.legal_moves())
    tree2.expand(board.legal_moves())

    while board.who_won() == 2:
        if board.player == t1:
            move = treeAI(board, tree1, iterations, value, c1).last_move

            tree1 = tree1.move(move)
            tree2 = tree2.move(move)

            board.move(*move)
        else:
            move = treeAI(board, tree2, iterations, value, c2).last_move

            tree1 = tree1.move(move)
            tree2 = tree2.move(move)

            board.move(*move)

    winner = board.who_won()
    return winner * t1


def play_human():
    m, n, k = map(int, input('Choose m, n, and k: ').split())
    iterations = int(input('Choose AI Iterations: '))
    human = random.choice([1, -1])

    board = mnk.Board(m, n, k)

    root = MCTree()
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
            root = treeAI(board, root, iterations, 'n', 2)
            board.move(*root.last_move)
        print(board)
        winner = board.who_won()

        if winner == human:
            print('You win!')
        elif winner == -human:
            print('You lose :(')
        elif winner == 0:
            print('You tie.')


def treeAI(board, tree, iterations, value, c):
    for i in range(iterations):
        tree.search(board, c)

    return tree.best_child(value)


def n_rollouts(board):
    avg = 0
    n = 1

    for _ in range(n):
        avg += rollout(board)
    return avg/n


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


class QLTree:
    def __init__(self, last_move=None):
        self.last_move = last_move
        self.q = 0
        self.n = 0
        self.children = []
        self.isLeaf = True

    def search(self, board, c):
        if self.isLeaf:
            winner = n_rollouts(board)
            if board.who_won() == 2:
                self.expand(board.legal_moves())
            self.n += 1
            self.q += winner * -board.player
        else:
            next_state = self.max_child(c)
            board.move(*next_state.last_move)
            next_state.search(board, c)
            board.undo_move()

            target = -max(self.children, key=lambda child: child.q).q
            self.n += 1
            self.q = self.q + (1 / self.n) * (target - self.q)

    def move(self, move):
        for child in self.children:
            if child.last_move == move:
                return child

    def expand(self, legal_moves):
        for move in legal_moves:
            self.children.append(QLTree(move))
        self.isLeaf = False

    def UCT(self, N, c):
        return self.q + c * math.sqrt(math.log(N + 1) / (self.n + 1))

    def max_child(self, c):
        return max(self.children, key=lambda child: child.UCT(self.n, c))

    def best_child(self, value):
        if value == 'q':
            return max(self.children, key=lambda child: child.q)
        else:
            return max(self.children, key=lambda child: child.n)


class DQLTree:
    def __init__(self, last_move=None):
        self.last_move = last_move
        self.q1 = 0
        self.q2 = 0
        self.n = 0
        self.children = []
        self.isLeaf = True

    def search(self, board, c):
        if self.isLeaf:
            if board.who_won() == 2:
                self.expand(board.legal_moves())

            winner = n_rollouts(board)

            self.n += 1
            if random.choice([True, False]):
                self.q1 += winner * -board.player
            else:
                self.q2 += winner * -board.player
        else:
            next_state = self.max_child(c)
            board.move(*next_state.last_move)
            next_state.search(board, c)
            board.undo_move()

            self.n += 1
            if random.choice([True, False]):
                target1 = -max(self.children, key=lambda child: child.q1).q2
                self.q1 = self.q1 + (2 / self.n) * (target1 - self.q1)
            else:
                target2 = -max(self.children, key=lambda child: child.q2).q1
                self.q2 = self.q2 + (2 / self.n) * (target2 - self.q2)

    def move(self, move):
        for child in self.children:
            if child.last_move == move:
                return child

    def expand(self, legal_moves):
        for move in legal_moves:
            self.children.append(DQLTree(move))
        self.isLeaf = False

    def UCT(self, N, c):
        return (self.q1 + self.q2)/2 + c * math.sqrt(math.log(N + 1) / (self.n + 1))

    def max_child(self, c):
        return max(self.children, key=lambda child: child.UCT(self.n, c))

    def best_child(self, value):
        if value == 'q':
            return max(self.children, key=lambda child: (child.q1 + child.q2)/2)
        else:
            return max(self.children, key=lambda child: child.n)


class MCTree:
    def __init__(self, last_move=None):
        self.last_move = last_move
        self.q = 0
        self.n = 0
        self.children = []
        self.isLeaf = True

    def search(self, board, c):
        if self.isLeaf:
            winner = n_rollouts(board)
            if board.who_won() == 2:
                self.expand(board.legal_moves())
        else:
            next_state = self.max_child(c)
            board.move(*next_state.last_move)
            winner = next_state.search(board, c)
            board.undo_move()

        self.n += 1
        self.q = self.q + (1 / self.n) * (winner * -board.player - self.q)

        return winner

    def move(self, move):
        for child in self.children:
            if child.last_move == move:
                return child

    def expand(self, legal_moves):
        for move in legal_moves:
            self.children.append(MCTree(move))
        self.isLeaf = False

    def UCT(self, N, c):
        return self.q + c * math.sqrt(math.log(N + 1) / (self.n + 1))

    def max_child(self, c):
        return max(self.children, key=lambda child: child.UCT(self.n, c))

    def best_child(self, value):
        if value == 'q':
            return max(self.children, key=lambda child: child.q)
        else:
            return max(self.children, key=lambda child: child.n)


if __name__ == '__main__':
    main()
