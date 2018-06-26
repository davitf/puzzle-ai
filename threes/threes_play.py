"""Game-specific information needed by play.py to play Threes interactively."""

import getkey

import threes.threes_util
from threes import threes_env


# Environment constructor.
Env = threes_env.ThreesGame

# Keys needed to play the game and corresponding actions.
KEYS_TO_ACTION = {
    getkey.keys.LEFT: threes.threes_util.DIRECTION_LEFT,
    getkey.keys.UP: threes.threes_util.DIRECTION_UP,
    getkey.keys.RIGHT: threes.threes_util.DIRECTION_RIGHT,
    getkey.keys.DOWN: threes.threes_util.DIRECTION_DOWN,
}

ACTION_NAMES = threes.threes_util.ACTION_NAMES
