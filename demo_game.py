import mnk

board = mnk.Board(5, 8, 3)

print(board)

board.move(0, 0)

print(board)

board.move(1, 1)

print(board)

board.move(2, 6)

print(board)

print(str(board.legal_moves()) + '\n')

print(board.board)

print(board.board.flatten())
