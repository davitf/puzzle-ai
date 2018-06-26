import random
import numpy as np

from threes import randomized_order_iter
import pytest


def get_items(iter, n):
    """Return the first n items from the iterator as a list."""
    output = []
    if not n:
        return output

    for item in iter:
        output.append(item)
        if len(output) == n:
            break

    return output


def check_same_contents(a, b):
    assert sorted(a) == sorted(b)


@pytest.mark.parametrize("input", [([1, 2, 3, 4, 5]), ([10, 10, 10, 10, 1])])
def test_randomized_order_iter(input):
    it = randomized_order_iter.randomized_order_iter(input)
    output = get_items(it, len(input) * 20)

    for i in range(0, len(output), len(input)):
        assert sorted(input) == sorted(output[i : i + len(input)])

    # Ensure that some shuffling has happened.
    assert len(output) == len(input * 20)  # Double-checking the test.
    assert output != input * 20


@pytest.mark.parametrize(
    "rand", [random, random.Random(2), np.random, np.random.RandomState(2)]
)
def test_random_instances(rand):
    input = [1, 2, 3, 4, 5]

    it = randomized_order_iter.randomized_order_iter(input, rand=rand)
    output = get_items(it, len(input) * 20)

    for i in range(0, len(output), len(input)):
        assert sorted(input) == sorted(output[i : i + len(input)])

    # Ensure that some shuffling has happened.
    assert len(output) == len(input * 20)  # Double-checking the test.
    assert output != input * 20
