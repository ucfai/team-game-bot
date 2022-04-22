#app.py
from pickle import GLOBAL
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import mnk, mcts_demo

# Libraries for SocketIO (Live Connection with Server)
from flask import render_template, session
from flask_socketio import SocketIO


app = Flask(__name__)
socketio = SocketIO(app)
#CORS(app)

# ======================================================== #
#                    Auxiliary Functions                   #
# ======================================================== #

def unpack(dimensions):
    """Given a string of the form "nxm", returns a tuple (n, m).

    Args:
        dimensions (str): A string of the form "nxm".

    Returns:
        typle: A tuple (n, m).
    """
    size = str(dimensions)
    return tuple(map(int, size.split('x')))

def toClientMatrix(board):
    """Given a board object, return its matrix representation
    to be interpreted by the client. (1 counter-clockwise rotation)
    of the raw board matrix. (1 = 'X', -1 = 'O', 0 = ' ')

    Args:
        board (mnk.Board): A board object.

    Returns:
        np.ndarray: Matrix representation of the board.
    """
    return np.rot90(board.board).tolist()


# ======================================================== #
#                       Main Routing                       #
# ======================================================== #

@app.route('/')
def index():
    return send_from_directory('../frontend/', 'index.html')

@app.route('/<webpage>')
def serve(webpage):
    return send_from_directory('../frontend/', webpage)


# ======================================================== #
#                     Get Boards (APIs)                    #
# ======================================================== #
    
@app.route('/board/random/<dimensions>/')
def random_nxm(dimensions):
    """Return an random matrix with 1s, 0s, and -1s  of the
    given dimensions. <dimensions> = string of the form "nxm".
    """
    size = unpack(dimensions)

    # Return (n,m) matrix with random numbers [-1, 0, 1]
    # representing the board state
    matrix = np.random.randint(-1, 2, size=size)

    # Convert matrix to a python list
    matrix = matrix.tolist()

    # Return the matrix as a JSON object, with name "board"
    return jsonify(board=matrix)


@app.route('/board/empty/<dimensions>/')
def empty_nxm(dimensions):
    """Return an empty matrix of the given dimensions.
    <dimensions> = string of the form "nxm".
    """
    size = unpack(dimensions)
    return jsonify(board=np.zeros(size).tolist())


# ======================================================== #
#                 User Events (Web Sockets)                #
# ======================================================== #

@socketio.on("connection")
def new_connection(json):
    print("New Connection")
    # route = json['route']
    # print(route)
    
    # Start a new game (7x7x3 Default)
    session["board"] = mnk.Board(7, 3, 3)

    #socketio.emit("board_update", session["board"].board.tolist())

    print(session["board"].board)

@socketio.on("new_game")
def new_game(json):
    print("New Game")

    print(json['k'])
    # { m: m, n:n, k: k }
    m = int(json['m'])
    n = int(json['n'])
    k = int(json['k'])

    # Start a new game
    session["board"] = mnk.Board(m, n, k)
    print(session["board"].board)
    socketio.emit("board_update", session["board"].board.tolist())


@socketio.on("user_move")
def user_move(json):
    board = session["board"]
    iterations = 1000 # Hard Coded for now
    # (TODO: Change iterations to user input. Send as JSON from client)

    # Get the user's move. Json = {i : row, j : column}
    i = json["i"] 
    j = json["j"]

    print("User Move:", i, j)
    print("Board Max:", board.m, board.n)
    # ======================================================== #
    # Note: the moves are somewhat messed up, because the board is
    #       represented as a matrix differently in the server and
    #       client. To fix it we do the following:

    # 1. Invert i <-> j (Could have done this above, but it's a bit more clear this way)
    # i, j = j + 1, board.m - i

    # 2 Invert the board vertically (i.e. invert j) 
    #j = board.m - j - 1
    #
    # End of Note :)
    # ======================================================== #
    print("User Move (Change):", i, j)
    print("Board Max:", board.m, board.n)
    # Make the move on the board
    board.move(i, j) # <---- 
    print(board)
    print(board.board)
    print(toClientMatrix(board))

    # Check user's move for win
    if board.who_won() != 2:
        # User won
        socketio.emit("win", 1)
        return

    # Get move from MCTS
    print('AI is thinking')
    root = mcts_demo.Node()
    if root.isLeaf:
        root.expand(board.legal_moves())

    root = mcts_demo.AI(board, root, iterations)
    board.move(*root.last_move)

    # Check AI's move for win
    if board.who_won() != 2:
        socketio.emit("win", 0)

    socketio.emit("board_update", board.board.tolist())
        
    # Send the board to the client
    # message = {
    #     "board": toClientMatrix(board),
    #     "who_won": board.who_won()
    # }

    


# ======================================================== #
#                       Start Server                       #
# ======================================================== #
if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app)


