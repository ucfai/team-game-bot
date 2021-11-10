from agent import Agent
import mnk
import tensorflow as tf

board = mnk.Board(3, 3, 3, form="multiplanar")
model = tf.keras.models.load_model('menagerie/6000')

print("\n\n" + str(board))
current_player = input("\nWho plays first (Me/AI)? ")
ai_side = [-1, 1][current_player == "AI"]
agent = Agent(board, model, ai_side, training=False)

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
        agent.action(0)
        current_player = "Me"

    print(board)

if board.who_won() == 0:
    print("Tie Game!")
elif current_player == "Me":
    print("AI Wins!")
else:
    print("You Win!")
