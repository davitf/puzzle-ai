"""An implementation of the Threes puzzle game as an OpenAI Gym environment.

See the basic mechanics at https://en.wikipedia.org/wiki/Threes .

The details of how the future tiles are chosen come from
https://toucharcade.com/community/threads/218248/page-27#post-3140044
with a few differences on how special future cards work, based on analyzing the
source of the web-based version at http://play.threesgame.com/ .

Differences in scoring:

The original game simply adds up the total value of all the tiles in the board
when a game ends. Since here we need to compute a reward for each move, the
straightforward approach would be to define the reward of a move as the
difference between the total value of the board before and after it, which is
equivalent to (value of tiles created from merging - value of the tiles which
were merged + value of the new added tile).

However, as the added tile is chosen randomly outside of the agent's control,
I chose to ignore its value when computing the step reward, to reduce variance
during learning. Thus the scoring in this version comes only from the merges
performed by the agent, and will be lower than the original game.
"""
from __future__ import division, print_function

import gym
import gym.utils.seeding
import numpy as np

from threes import threes_render
from threes.randomized_order_iter import randomized_order_iter

# The game state provided by the environment is a one-dimensional array of ints,
# containing the type of the current tiles in all spaces in the board, plus the
# type of the next tile.
#
# I.e. for a 4x4 board:
#
#    a  b  c  d
#    e  f  g  h
#    i  j  k  l
#    m  n  o  p
#
#    NEXT PIECE: q
#
# The state is [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q]
#
# When the next tile is a "special" one (i.e. not 1, 2 or 3), the player is
# given up to three possibilities as to what it may be; in this case, the state
# contains only the highest possibility, and the actual tile will be it or one
# of the two immediately lower ones, but always at least a 6 (i.e. if "q" in the
# state above is TILE_12, the actual tile might be TILE_6 or TILE_12; for a
# TILE_192, the actual one might be TILE_48, TILE_96 or TILE_192).

from threes.threes_util import (
    TILE_1,
    TILE_2,
    TILE_3,
    TILE_24,
    TILE_48,
    NUM_FEATURES,
    NUM_ACTIONS,
    is_game_over,
    move_board,
    future_tile_possibilities,
)


# How often a special tile (not 1, 2 or 3) appears: one per 21 moves.
SPECIAL_TILE_FREQUENCY = 21

# How many tiles are placed in the board when the game starts.
INITIAL_TILES_IN_BOARD = 9


class ThreesGame(gym.Env):
    """OpenAI Gym environment implementing the Threes puzzle game."""

    metadata = {"render.modes": ["ansi", "human"]}
    action_space = gym.spaces.Discrete(NUM_ACTIONS)

    # observation_space depends on board_size, which is set for each instance.
    # observation_space = gym.spaces.MultiDiscrete([NUM_FEATURES] * STATE_SIZE)

    def __init__(self, board_size=4, high_tiles=None):
        """Initialize the environment.

        :param board_size: How many tiles on each side of the board.
        :param high_tiles: Optional list of possible tiles to add to the board
            when the game starts (always on the top left corner). One value will
            be chosen randomly for each game. This can also be a single value,
            which will be used for all games.
        """
        self.board_size = board_size
        # Convert high_tile to a one-element list if it is not a list.
        self.high_tiles = (
            high_tiles if hasattr(high_tiles, "__len__") else [high_tiles]
        )

        self.board_area = board_size * board_size  # Number of tiles in the board.

        self.state_size = self.board_area + 1  # All tiles in the board + next tile.
        self.observation_space = gym.spaces.MultiDiscrete(
            [NUM_FEATURES] * self.state_size
        )

        self.state = np.zeros(self.state_size, dtype="int")

        self._create_arrays()
        self.seed(None)
        self.reset()

    def _create_arrays(self):
        # Create the board and future_tile arrays, which share a base with state.

        # The contents of the board are stored as a flattened array in the
        # first elements of the state.
        self.board = self.state[: self.board_area].reshape(
            (self.board_size, self.board_size)
        )

        # The future tile is stores in the last element of the state.
        self._future_tile_arr = self.state[self.board_area :]
        assert len(self._future_tile_arr) == 1

    def __setstate__(self, class_state):
        # Called when unpickling. Needed for calling _create_arrays after
        # self.state is replaced with the saved version.

        # Restore instance attributes.
        self.__dict__.update(class_state)

        self._create_arrays()

    @property
    def future_tile(self):
        return self._future_tile_arr[0]

    @future_tile.setter
    def future_tile(self, future_tile):
        self._future_tile_arr[0] = future_tile

    def reset(self):
        # Initialize the next-tile queues.
        self.next_tile_queue = randomized_order_iter(
            [TILE_1, TILE_2, TILE_3] * 4, self.np_random
        )

        # A queue is also used for deciding when to output special tiles
        # (in play.threesgame.com).
        self.special_queue = randomized_order_iter(
            [True] + [False] * (SPECIAL_TILE_FREQUENCY - 1), self.np_random
        )

        self._init_board()
        self._choose_future_tile_in_state()
        self.total_score = 0

        return self.state

    def render(self, mode, close=False):
        if close:
            return

        rendered = threes_render.render_ansi(self.board, self.future_tile)

        if mode == "ansi":
            return rendered
        elif mode == "human":
            print(rendered)

    def close(self):
        pass

    def seed(self, seed):
        self.np_random, seed = gym.utils.seeding.np_random(seed)

    def _init_board(self):
        """Creates a new-game board with the pre-populated tiles."""
        board = [0] * (self.board_area - INITIAL_TILES_IN_BOARD)

        for _ in range(INITIAL_TILES_IN_BOARD - bool(self.high_tiles)):
            board.append(next(self.next_tile_queue))
        self.np_random.shuffle(board)

        if self.high_tiles:
            high_tile = self.np_random.choice(self.high_tiles)
            board = [high_tile] + board

        self.state[: self.board_area] = board

    def _choose_future_tile_in_state(self):
        highest_tile = np.max(self.board)
        is_special = highest_tile >= TILE_48 and next(self.special_queue)
        if not is_special:
            self.future_tile = next(self.next_tile_queue)
            return

        max_special_tile = highest_tile - 3

        # Here we choose only the high limit of the next tile.
        if max_special_tile <= TILE_24:
            self.future_tile = max_special_tile
        else:
            self.future_tile = self.np_random.randint(TILE_24, max_special_tile + 1)

    def choose_tile_to_add(self):
        possible_tiles = future_tile_possibilities(self.future_tile)
        if len(possible_tiles) == 1:
            return possible_tiles[0]
        return self.np_random.choice(possible_tiles)

    def step(self, action):
        """Performs a step.

        Raises: IllegalMoveError
        """
        # Perform the board movement.
        tile_to_add = self.choose_tile_to_add()
        score_delta = move_board(self.board, action, tile_to_add, self.np_random)
        self.total_score += score_delta

        # Choose the future tile to be included in the state.
        self._choose_future_tile_in_state()

        game_over = is_game_over(self.board)
        return (self.state, score_delta, game_over, {})
