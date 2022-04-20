#app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

import numpy
print (np.__version__)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('../frontend/', 'index.html')

@app.route('/<webpage>')
def serve(webpage):
    return send_from_directory('../frontend/', webpage)
    
@app.route('/board/<dimensions>/')
def respond_nxm(dimensions):

    size = str(dimensions)

    # Convert size "nxm" to a tuple
    size = tuple(map(int, size.split('x')))

    # Return (n,m) matrix with random numbers [-1, 0, 1]
    # representing the board state
    matrix = np.random.randint(-1, 2, size=size)

    # Convert matrix to a python list
    matrix = matrix.tolist()

    # Return the matrix as a JSON object, with name "board"
    return jsonify(board=matrix)
    
if __name__ == "__main__":
    app.run()
    




