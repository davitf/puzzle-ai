"""An iterator which outputs items from a list endlessly in a random order.

This is equivalent to::

    def randomized_order_iter(contents, rand=np.random):
        while True:
            rand.shuffle(contents)
            yield from contents

but can be pickled and restored, unlike the generator-based version above.
"""

import numpy as np


class randomized_order_iter(object):
    """Iterator which outputs items from a list endlessly in a random order."""

    def __init__(self, contents, rand=np.random):
        """
        :param contents: The list of items which will be output one at a time.
            The list will be shuffled in-place, so should not be shared with
            other instances.
        :param rand: A `numpy.random.RandomState` or `random.Random` instance
            to use for shuffling the items.
        """
        self.contents = contents
        self.rand = rand
        self.pos = len(contents) - 1  # shuffle next time

    def __iter__(self):
        return self

    def __next__(self):
        self.pos += 1
        if self.pos == len(self.contents):  # All items output, must reshuffle.
            self.rand.shuffle(self.contents)
            self.pos = 0

        return self.contents[self.pos]
