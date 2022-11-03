from agent import Agent
import mnk
import tensorflow as tf
import model
import sys

board = mnk.Board(3, 3, 3)

#assert len(sys.argv) == 2, "Please specify which model you would like to play against (ex: python3 play.py models/PedrosModel).\n Tab complete works!"
#model = model.Model((3, 3, 3), sys.argv[1])

model = model.Model((3, 3, 3), "new_model")

print("\n\n" + str(board))
current_player = input("\nWho plays first (Me/AI)? ")
ai_side = [-1, 1][current_player == "AI"]
agent = Agent(model, ai_side)

while board.who_won() == 2:
    if current_player == 'Me':
        played = False
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
        board.move(*agent.action(board))
        current_player = "Me"

    print(board)

if board.who_won() == 0:
    print("Tie Game!")
elif current_player == "Me":
    print("AI Wins!")
else:
    print("You Win!")
