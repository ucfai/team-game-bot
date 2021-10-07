import mnk
board = mnk.Board(3, 3, 2)

print("\n\n" + str(board))
current_player = input("\nWho plays first (Me/AI)? ")

while board.who_won() == "Ongoing Game":
    played = False

    if current_player == 'Me':
        while not played:
            move = int(input("What is your move (1-9)? "))
            x = (move - 1) % 3
            y = 2 - ((move - 1) // 3)

            try:
                board.move(x, y)
                played = True
            except:
                print("Invalid move! Try again")

        current_player = "AI"
    else:
        board.play_ai_move()
        current_player = "Me"
    print(board)

if board.who_won() == 'Tie':
    print("Tie Game!")
elif current_player == "Me":
    print("AI Wins!")
else:
    print("You Win!")
