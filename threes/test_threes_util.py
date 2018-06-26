import numpy as np
import pytest

from threes import threes_util

# A series of sample lines, with the results of moving them left or right.
SAMPLE_LINES_BOTH_DIRECTIONS = [
    # Move spaces.
    ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]),
    ([1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]),
    ([0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]),
    ([4, 3, 0, 4], [4, 3, 4, 0], [0, 4, 3, 4]),
    ([2, 0, 0, 3], [2, 0, 3, 0], [0, 2, 0, 3]),
    ([2, 0, 1, 0], [2, 1, 0, 0], [0, 2, 0, 1]),
    # Merge 1s and 2s.
    ([1, 2, 4, 3], [3, 4, 3, 0], [0, 3, 4, 3]),
    ([4, 2, 1, 3], [4, 3, 3, 0], [0, 4, 3, 3]),
    ([4, 3, 2, 1], [4, 3, 3, 0], [0, 4, 3, 3]),
    ([1, 2, 1, 2], [3, 1, 2, 0], [0, 1, 2, 3]),
    ([1, 2, 1, 5], [3, 1, 5, 0], [0, 1, 3, 5]),
    ([0, 1, 2, 3], [1, 2, 3, 0], [0, 0, 3, 3]),
    # Merge two equal 3+ pieces.
    ([2, 2, 3, 3], [2, 2, 4, 0], [0, 2, 2, 4]),
    ([4, 4, 0, 1], [5, 0, 1, 0], [0, 4, 4, 1]),
    ([2, 8, 8, 8], [2, 9, 8, 0], [0, 2, 8, 9]),
    # Two 1s or 2s cannot be merged.
    ([2, 2, 4, 3], [2, 2, 4, 3], [2, 2, 4, 3]),
    ([4, 1, 1, 3], [4, 1, 1, 3], [4, 1, 1, 3]),
    # Non-contiguous pieces cannot be merged.
    ([2, 4, 2, 4], [2, 4, 2, 4], [2, 4, 2, 4]),
    ([2, 3, 1, 4], [2, 3, 1, 4], [2, 3, 1, 4]),
]

# move_line only computes the result of moving a line to the left. For a
# move to the right, we must invert the input and output. Here we convert
# the test data set above into pairs of expected (input, output) pairs.
move_line_testdata = []
for input, moved_left, moved_right in SAMPLE_LINES_BOTH_DIRECTIONS:
    move_line_testdata.append((input, moved_left))
    move_line_testdata.append((input[::-1], moved_right[::-1]))


@pytest.mark.parametrize("original, moved_left", move_line_testdata)
def test_is_line_movable_left(original, moved_left):
    is_movable = original != moved_left
    line = np.array(original)
    assert threes_util.is_line_movable_left(line) == is_movable
    # Check that the argument was not modified in-place.
    assert line.tolist() == original


@pytest.mark.parametrize("input, output", move_line_testdata)
def test_move_line_left(input, output):
    line = np.array(input)
    moved, score_delta = threes_util.move_line_left(line)
    assert np.all(line == output)
    assert moved == (input != output)
    assert score_delta == (
        threes_util.total_score(output) - threes_util.total_score(input)
    )


# A sample Threes board, with the result of moving it in each direction.
# In the result boards, the possible spots where a new piece can be placed are
# marked with -1 values.
# This board cannot be moved up.
SAMPLE_BOARD = [[3, 4, 7, 4], [1, 2, 3, 0], [4, 5, 0, 0], [1, 1, 0, 0]]

SAMPLE_BOARD_LEFT = [[3, 4, 7, 4], [3, 3, 0, -1], [4, 5, 0, 0], [1, 1, 0, 0]]

SAMPLE_BOARD_RIGHT = [[3, 4, 7, 4], [-1, 1, 2, 3], [-1, 4, 5, 0], [-1, 1, 1, 0]]

SAMPLE_BOARD_DOWN = [[3, 4, -1, -1], [1, 2, 7, 4], [4, 5, 3, 0], [1, 1, 0, 0]]


class FakeRandom(object):
    """A fake np.random implementation which expects a .choice() call."""

    def __init__(self, max, return_value):
        self.max = max
        self.return_value = return_value

    def choice(self, max):
        assert max == self.max
        return self.return_value


def generate_next_piece_boards(board, new_piece):
    possible_new_piece_places = np.argwhere(board == -1)
    new_boards = []

    for new_piece_location in possible_new_piece_places:
        new_board = board.copy()
        new_board[board == -1] = 0
        new_board[tuple(new_piece_location)] = new_piece
        new_boards.append(new_board.tolist())

    return new_boards


@pytest.mark.parametrize(
    "orig_board, expected_board, direction",
    [
        (SAMPLE_BOARD, SAMPLE_BOARD_LEFT, threes_util.DIRECTION_LEFT),
        (SAMPLE_BOARD, SAMPLE_BOARD_RIGHT, threes_util.DIRECTION_RIGHT),
        (SAMPLE_BOARD, SAMPLE_BOARD_DOWN, threes_util.DIRECTION_DOWN),
    ],
)
def test_move_board(orig_board, expected_board, direction):
    orig_board = np.array(orig_board)
    expected_board = np.array(expected_board)

    # Generate the possible new boards. In each one, one of the -1s is replaced
    # by the new piece (which we're hardcoding to 10), and the others by 0s
    # (empty spaces).
    num_new_boards = np.sum(expected_board == -1)
    expected_new_boards = generate_next_piece_boards(expected_board, 10)
    assert num_new_boards == len(expected_new_boards)

    # If there are no new boards, this is an illegal move, which is tested
    # separately.
    assert num_new_boards

    # Run move_board with all the possible random values, so that it should
    # generate all the possibilities (in an unknown order).
    generated_new_boards = []
    for i in range(num_new_boards):
        moved_board = orig_board.copy()
        score_delta = threes_util.move_board(
            moved_board, direction, 10, np_rand=FakeRandom(num_new_boards, i)
        )
        assert score_delta == (
            threes_util.total_score(expected_board)
            - threes_util.total_score(orig_board)
        )

        generated_new_boards.append(moved_board.tolist())

    # Check that the generated boards match the expected ones. The lists are
    # sorted so that differences in ordering are ignored.
    # (as in unittest.TestCase.assertCountEqual , which doesn't have a
    # py.test equivalent).
    assert sorted(expected_new_boards) == sorted(generated_new_boards)


def test_move_board_illegal_action():
    board = np.array(SAMPLE_BOARD)

    with pytest.raises(threes_util.IllegalMoveError):
        delta = threes_util.move_board(
            board, threes_util.DIRECTION_UP, 10, np_rand=FakeRandom(1, 0)
        )

    # Ensure that an illegal move does not change the input array.
    assert np.all(board == SAMPLE_BOARD)
