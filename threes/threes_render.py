"""Renders a Threes board into an ANSI string to be printed to the console."""

from colorama import Fore, Back, Style

from threes import threes_util

# Name and colors to use for printing each possible tile.
TILE_NAMES = ["      ", " -1-  ", " -2-  "] + [
    "{:^5} ".format(3 * 2 ** i) for i in range(13)
]
BACK_COLORS = [Back.BLACK, Back.BLUE, Back.RED] + [Back.WHITE] * 20
FRONT_COLORS = [Fore.WHITE, Fore.WHITE, Fore.WHITE] + [Fore.BLACK] * 20


def render_ansi(board, future_tile):
    """Generate a string for printing a Threes game state into a console."""
    LEFT_MARGIN = " " * 10

    to_write = "\n\n"
    possible_tiles = threes_util.future_tile_possibilities(future_tile)
    tile_name_line = padding_line = "    " * (3 - len(possible_tiles))

    # Render the "Next:"  line.
    for tile in possible_tiles:
        back = BACK_COLORS[tile]
        front = FRONT_COLORS[tile]
        padding_line += "  " + back + front + " " * 6 + Style.RESET_ALL
        tile_name_line += "  " + back + front + TILE_NAMES[tile] + Style.RESET_ALL

    to_write += LEFT_MARGIN + "      " + padding_line + "\n"
    to_write += LEFT_MARGIN + "Next: " + tile_name_line + "\n"
    to_write += LEFT_MARGIN + "      " + padding_line + "\n\n\n"

    # Render the lines of the board, one at a time (each board line takes
    # multiple console lines).
    for i, line in enumerate(board):
        padding_line = LEFT_MARGIN
        tiles = LEFT_MARGIN
        for j, tile in enumerate(line):
            back = BACK_COLORS[tile]
            front = FRONT_COLORS[tile]
            padding_line += back + front + " " * 6 + Style.RESET_ALL + "  "
            tiles += back + front + TILE_NAMES[tile] + Style.RESET_ALL + "  "
        to_write += padding_line + "\n" + tiles + "\n" + padding_line + "\n\n"

    return to_write
