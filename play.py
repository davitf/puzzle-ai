import time
import pickle

import getkey

import threes.threes_util
from common import env_common

# TODO: choose environment to play.
import threes.threes_play as game_play

kwargs = {"high_pieces": threes.threes_util.PIECE_768}

game = game_play.Env(**kwargs)

saved = []

while True:
    score = delta = 0
    n_moves = 0
    game_over = False
    state = game.reset()

    while True:
        print("Score: %d (%d)  Moves: %d   State: %s" % (score, delta, n_moves, state))
        game.render("human")

        if game_over:
            print("Game over")
            break

        key = getkey.getkey()

        if key == getkey.keys.ESC:
            exit()
        elif key == getkey.keys.BACKSPACE:
            if not saved:
                print("No data to restore")
                continue
            print("Restoring")
            (game, score, n_moves) = pickle.loads(saved.pop(-1))
            state = game.state
            continue

        elif key not in game_play.KEYS_TO_ACTION:
            print("bad key", key)
            continue
        else:
            action = game_play.KEYS_TO_ACTION[key]

        try:
            saved.append(pickle.dumps((game, score, n_moves), protocol=4))
            if len(saved) > 1000:
                saved = saved[10:]

            state, delta, game_over, internals = game.step(action)
            score += delta
            n_moves += 1
            if "intermediate_boards" in internals:
                for intermediate_board in internals["intermediate_boards"]:
                    game.render("human", intermediate_board=intermediate_board)
                    time.sleep(0.5)

        except env_common.IllegalMoveError as e:
            print(repr(e))
            continue
