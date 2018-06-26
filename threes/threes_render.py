"""Renders a Threes board into an ANSI string to be printed to the console."""

from colorama import Fore, Back, Style

from threes import threes_util

# Name and colors to use for printing each possible piece.
PIECE_NAMES = ["      ", " -1-  ", " -2-  "] + [
    "{:^5} ".format(3 * 2 ** i) for i in range(13)
]
BACK_COLORS = [Back.BLACK, Back.BLUE, Back.RED] + [Back.WHITE] * 20
FRONT_COLORS = [Fore.WHITE, Fore.WHITE, Fore.WHITE] + [Fore.BLACK] * 20


def render_ansi(board, future_piece):
    """Generate a string for printing a Threes game state into a console."""
    LEFT_MARGIN = " " * 10

    to_write = "\n\n"
    possible_pieces = threes_util.future_piece_possibilities(future_piece)
    piece_name_line = padding_line = "    " * (3 - len(possible_pieces))

    # Render the "Next:"  line.
    for piece in possible_pieces:
        back = BACK_COLORS[piece]
        front = FRONT_COLORS[piece]
        padding_line += "  " + back + front + " " * 6 + Style.RESET_ALL
        piece_name_line += "  " + back + front + PIECE_NAMES[piece] + Style.RESET_ALL

    to_write += LEFT_MARGIN + "      " + padding_line + "\n"
    to_write += LEFT_MARGIN + "Next: " + piece_name_line + "\n"
    to_write += LEFT_MARGIN + "      " + padding_line + "\n\n\n"

    # Render the lines of the board, one at a time (each board line takes
    # multiple console lines).
    for i, line in enumerate(board):
        padding_line = LEFT_MARGIN
        pieces = LEFT_MARGIN
        for j, piece in enumerate(line):
            back = BACK_COLORS[piece]
            front = FRONT_COLORS[piece]
            padding_line += back + front + " " * 6 + Style.RESET_ALL + "  "
            pieces += back + front + PIECE_NAMES[piece] + Style.RESET_ALL + "  "
        to_write += padding_line + "\n" + pieces + "\n" + padding_line + "\n\n"

    return to_write
