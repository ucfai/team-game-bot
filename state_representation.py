import numpy as np


# Reshapes board into the form needed for the model
def get_input_rep(self, board, form):
    m, n = np.shape(board)

    if form == "flatten":
        return np.copy(board.reshape(1, 1, m * n))

    elif form == "planar":
        return np.copy(board.reshape(1, m, n, 1))

    elif form == "multiplanar":
        board_planes = np.zeros((m, n, 2), dtype=int)
        for i in range(m):
            for j in range(n):
                if board[i][j] == 1:
                    board_planes[i][j][0] = 1
                elif board[i][j] == -1:
                    board_planes[i][j][1] = 1
        return np.copy(board_planes.reshape(1, m, n, 2))

    elif form == "multiplanar-turnflipped":
        board_planes = np.zeros((m, n, 2), dtype=int)
        for i in range(m):
            for j in range(n):
                if self.board[i][j] == self.player:
                    board_planes[i][j][0] = 1
                elif self.board[i][j] == -1 * self.player:
                    board_planes[i][j][1] = 1
        return np.copy(board_planes.reshape(1, m, n, 2))
