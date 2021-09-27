import unittest
import mnk

class TestMnk(unittest.TestCase):
    def setUp(self):
        self.board = mnk.Board(3, 3, 2)
    def test_repr(self):
        self.assertEqual(repr(self.board), '_|_|_\n'*3)
    def test_move(self):
        self.board.move(0, 0)
        self.board.move(1, 2)
        self.board.move(2, 1)
        self.assertEqual(repr(self.board),
            '_|O|_\n'
            '_|_|X\n'
            'X|_|_\n')
    def test_undo_move(self):
        self.board.move(0, 0)
        self.board.undo_move(0, 0)
        self.assertEqual(repr(self.board), '_|_|_\n'*3)
    def test_generate_moves(self):
        self.assertEqual(len(self.board.generate_moves()), 9)
        self.board.move(0, 0)
        self.assertEqual(len(self.board.generate_moves()), 8)
    def test_lost(self):
        self.board.move(1, 1)
        self.board.move(2, 2)
        self.assertFalse(self.board.lost())
        self.board.move(0, 0)
        self.assertTrue(self.board.lost())
        self.board.undo_move(0, 0)
        self.board.move(1, 0)
        self.assertTrue(self.board.lost())
        self.board.undo_move(1, 0)
        self.board.move(2, 0)
        self.assertTrue(self.board.lost())
        self.board.undo_move(2, 0)
        self.board.move(2, 1)
        self.assertTrue(self.board.lost())

if __name__ == '__main__':
    unittest.main()
