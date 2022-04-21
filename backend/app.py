#app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import mnk, mcts_demo

print (np.__version__)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('../frontend/', 'index.html')

@app.route('/<webpage>')
def serve(webpage):
    return send_from_directory('../frontend/', webpage)

# Converts size "nxm" to a tuple
def unpack(dimensions):
    size = str(dimensions)
    return tuple(map(int, size.split('x')))
    
@app.route('/board/random/<dimensions>/')
def random_nxm(dimensions):
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
    size = unpack(dimensions)
    return jsonify(board=np.zeros(size).tolist())

@app.route('/play', methods=['POST'])
def play():
    currentBoard = request.json
    return ''

    
if __name__ == '__main__':
  app.run(debug=True)