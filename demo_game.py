import mnk

board = mnk.Board(3, 3, 2)

print(board)

for i in range(9):
    board.play_ai_move()
    print(board)
