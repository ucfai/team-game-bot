import unittest
import mnk

class TestMnk(unittest.TestCase):
    def setUp(self):
        self.board = mnk.Board(3, 3, 2)
    def test_str(self):
        self.assertEqual(str(self.board), '_|_|_\n'*3)
    def test_move(self):
        self.board.move(0, 0)
        self.board.move(1, 2)
        self.board.move(2, 1)
        self.assertEqual(str(self.board),
            '_|O|_\n'
            '_|_|X\n'
            'X|_|_\n')
    def test_undo_move(self):
        self.board.move(0, 0)
        self.board.undo_move(0, 0)
        self.assertEqual(str(self.board), '_|_|_\n'*3)
    def test_legal_moves(self):
        self.assertEqual(len(self.board.legal_moves()), 9)
        self.board.move(0, 0)
        self.assertEqual(len(self.board.legal_moves()), 8)
    def test_player_has_lost(self):
        self.board.move(1, 1)
        self.board.move(2, 2)
        self.assertFalse(self.board.player_has_lost())
        self.board.move(0, 0)
        self.assertTrue(self.board.player_has_lost())
        self.board.undo_move(0, 0)
        self.board.move(1, 0)
        self.assertTrue(self.board.player_has_lost())
        self.board.undo_move(1, 0)
        self.board.move(2, 0)
        self.assertTrue(self.board.player_has_lost())
        self.board.undo_move(2, 0)
        self.board.move(2, 1)
        self.assertTrue(self.board.player_has_lost())

if __name__ == '__main__':
    unittest.main()
