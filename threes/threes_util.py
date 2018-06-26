"""Basic constants and functions for Threes implementation and analysis."""

from typing import Tuple

import numpy as np

from common.env_common import IllegalMoveError

# The values representing each possible piece type in the state.
EMPTY_SPACE = 0
PIECE_1 = 1
PIECE_2 = 2
PIECE_3 = 3
PIECE_6 = 4
PIECE_12 = 5
PIECE_24 = 6
PIECE_48 = 7
PIECE_96 = 8
PIECE_192 = 9
PIECE_384 = 10
PIECE_768 = 11
PIECE_1536 = 12
PIECE_3072 = 13
PIECE_6144 = 14
PIECE_12288 = 15  # When this piece is created, the game ends automatically.

MAX_PIECE = PIECE_12288
NUM_FEATURES = MAX_PIECE + 1  # Possible values for each number in the state.

# Indicates possible places where a new piece can be added in the board.
# Will not appear in game states output by the environment, but is present
# in the boards returned by `move_board_preview`.
# (this is -100 instead of -1 to increase the chances of an IndexError happening
# if these incomplete boards are mistakenly used as real boards).
POSSIBLE_FUTURE_PIECE = -100

# The score for each piece on the final board:
# EMPTY_SPACE, PIECE_1 and PIECE_2: No points awarded.
# PIECE_3 (3): 3 = 3**1
# PIECE_6 (4): 9 == 3**2
# PIECE_12 (5): 27 == 3**3
# PIECE_24 (6): 81 == 3**4
# ...and so on.
SCORES = [0, 0, 0] + [3 ** i for i in range(1, 14)]

NUM_ACTIONS = 4  # Possible actions the user can take (the four directions).
DIRECTION_LEFT = 0
DIRECTION_UP = 1
DIRECTION_RIGHT = 2
DIRECTION_DOWN = 3

ACTION_NAMES = ["left", "up", "right", "down"]


def move_line_left(line, do_move=True) -> Tuple[bool, int]:
    """Move pieces towards the left. Return: (moved_bool, score_delta)

    `line` is modified in-place, unless `do_move` is False.

    :param line: The line to be moved, as a `list` or `np.ndarray`.
    :param do_move: Whether to actually perform the move. If False, the state is
        analyzed but `line` is not modified.
    :return: (moved_bool, score_delta), where moved_bool indicates whether the
        line was (or would be) moved, and score_delta indicates how many points
        are gained by merging pieces in the line.
    """
    for i in range(len(line) - 1):
        if line[i] == EMPTY_SPACE and line[i + 1] != EMPTY_SPACE:
            # Move everything after the empty space to the left.
            score_delta = 0
            if do_move:
                line[i:-1] = line[i + 1 :]
                line[-1] = EMPTY_SPACE

            break

        elif PIECE_1 <= line[i] <= PIECE_2 and line[i] + line[i + 1] == 3:
            # Merge a 1 and a 2 into a 3.
            score_delta = SCORES[PIECE_3]  # 1s and 2s are worth 0 points.
            if do_move:
                line[i] = PIECE_3
                line[i + 1 : -1] = line[i + 2 :]
                line[-1] = EMPTY_SPACE

            break

        elif PIECE_3 <= line[i] == line[i + 1]:
            score_delta = SCORES[line[i] + 1] - 2 * SCORES[line[i]]
            if do_move:
                line[i] += 1
                line[i + 1 : -1] = line[i + 2 :]
                line[-1] = EMPTY_SPACE

            break

    else:
        # Did not move
        return (False, 0)

    return (True, score_delta)


def is_line_movable_left(line):
    """True if line can be moved towards the beginning, False if it's stuck."""
    return move_line_left(line, False)[0]


def is_legal_action(board, direction: int):
    """Compute whether the board can be moved in the given direction.

    :param board: A 2-dimensional numpy array containing a board.
    :param direction: One of the valid actions.
    :return: True if the board can be moved in the given direction.
    """
    rotated = rotated_boards_view(board, direction)
    return any(is_line_movable_left(line) for line in rotated)


def is_game_over(board) -> bool:
    """True if there is no possible action in the board."""
    if PIECE_12288 in board:  # 12... piece
        return True

    if any(
        is_line_movable_left(line) or is_line_movable_left(line[::-1]) for line in board
    ) or any(
        is_line_movable_left(line) or is_line_movable_left(line[::-1])
        for line in board.T
    ):
        return False

    return True


def rotated_boards_view(board, direction: int):
    """Rotate the board(s) so that the given direction now points left.

    :param board: A 2-dimensional array (containing a single board) or
        3-dimensional array (containing several boards).
    :param direction: The direction which should be moved to the left.
    :return: A view of the board array in which `direction` in `board` now
        corresponds to the left direction. The view shares its base with the
        input array (i.e. changes to it will cause corresponding changes to
        the input).
    """

    # This function works for both single-board and multi-board arrays. Check
    # that the input array contains square boards, not 1-dimensional states.
    assert board.shape[-1] == board.shape[-2], board.shape
    if direction == 0:
        return board
    rotated = np.rot90(board, k=direction, axes=(-2, -1))
    assert rotated.base is board or rotated.base is board.base
    return rotated


def _move_board(board, direction: int) -> int:
    # Moves the board in-place.
    # Returns the score_delta and a list of the rows where a new piece can
    # be added in the last column.
    rotated = rotated_boards_view(board, direction)

    # Perform the board moves, store the lines that moved.
    total_score_delta = 0
    nextpiece_possible_rows = []
    for line in rotated:
        moved, score_delta = move_line_left(line)
        total_score_delta += score_delta
        if moved:
            nextpiece_possible_rows.append(line)

    if not nextpiece_possible_rows:
        raise IllegalMoveError

    return total_score_delta, nextpiece_possible_rows


def move_board(board, direction: int, piece_to_add: int, np_rand) -> int:
    """Move the pieces in the board in a given direction, adding a new piece.

    :param board: A 2-dimensional array containing a game board. The array
        will be modified in-place.
    :param direction: The direction to move the board in.
    :param piece_to_add: The new piece to add to the board.
    :param np_rand: A `numpy.random.RandomState` instance to use for choosing
        where to place the new piece.
    :return: The score gained by any merged pieces in the board.
    :raise IllegalMoveError: If the board cannot be moved in the given
        direction.
    """
    total_score_delta, nextpiece_possible_rows = _move_board(board, direction)

    # Add a new piece in one of the moved rows.
    row_to_add = np_rand.choice(len(nextpiece_possible_rows))
    nextpiece_possible_rows[row_to_add][-1] = piece_to_add

    return total_score_delta


def move_board_preview(board, direction: int) -> int:
    """Compute the consequences of moving the board in the given direction.

    :param board: A 2-dimensional array containing a game board. The array
        will not be modified.
    :param direction: The direction to move the board in.
    :return: (moved_board, score_delta), where:
        - moved_board is a new array with the result of moving the board in the
            given direction. The possible places where a new piece can appear will
            have the value `POSSIBLE_FUTURE_PIECE`.
        - score_delta is he score gained by any merged pieces in the board.
    :raise IllegalMoveError: If the board cannot be moved in the given
        direction.
    """
    """"""
    moved_board = board.copy()
    total_score_delta, nextpiece_possible_rows = _move_board(moved_board, direction)
    for row in nextpiece_possible_rows:
        row[-1] = POSSIBLE_FUTURE_PIECE

    return moved_board, total_score_delta


def future_piece_possibilities(piece_code):
    """Return the possible future pieces, from the piece value included in the state.
    """
    if piece_code <= PIECE_6:
        return [piece_code]
    return range(max(PIECE_6, piece_code - 2), piece_code + 1)


def total_score(input):
    """Return the total score value of a line or board."""
    # Ignore POSSIBLE_FUTURE_PIECE values in the input.
    return sum(SCORES[x] for x in np.array(input).flatten() if x >= 0)
