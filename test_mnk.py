import mnk

def test_str():
    board = mnk.Board(3, 3, 2)
    assert str(board) == '_|_|_\n'*3

def test_move():
    board = mnk.Board(3, 3, 2)
    board.move(0, 0)
    board.move(1, 2)
    board.move(2, 1)
    assert str(board) == (
        '_|O|_\n'
        '_|_|X\n'
        'X|_|_\n')

def test_undo_move():
    board = mnk.Board(3, 3, 2)
    board.move(0, 0)
    board.undo_move()
    assert str(board) == '_|_|_\n'*3

def test_legal_moves():
    board = mnk.Board(3, 3, 2)
    assert len(board.legal_moves()) == 9
    board.move(0, 0)
    assert len(board.legal_moves()) == 8

def test_player_has_lost():
    board = mnk.Board(3, 3, 2)
    board.move(1, 1)
    board.move(2, 2)
    assert not board.player_has_lost()
    board.move(0, 0)
    assert board.player_has_lost()
    board.undo_move()
    board.move(1, 0)
    assert board.player_has_lost()
    board.undo_move()
    board.move(2, 0)
    assert board.player_has_lost()
    board.undo_move()
    board.move(2, 1)
    assert board.player_has_lost()