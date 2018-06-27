"""Common utilities for neural network models."""

import numpy as np

_ARANGE = np.arange(4096)


def onehot_encode(input, num_values, dtype="float32", new_axis=False):
    """One-hot encode the input.

    :param input: A np.array with arbitrary format, such that
        np.all(0 <= input < num_features)
    :param num_values: The number of possible values for each feature of the
        input.
    :param dtype: The dtype of the output array.
    :param new_axis: If False, the output will have the same number of
        dimensions as the input, with the size of the last axis multiplied by
        `num_features` (e.g. for a 2-dimensional input,
        `input[a][b] == c` --> `output[a][b * num_features + c] == 1`).
        If True, the output will have one extra axis at the end
        (`input[a][b] == c` --> ``output[a][b][c] == 1`).
    :return: A np.array in which each of the features of the input has been
        transformed into `num_values` features, with the value corresponding to
        the feature set to 1 and the others set to 0.
    """
    original_sample_size = input.shape[-1]

    if new_axis:
        output_shape = input.shape + (num_values,)
    else:
        output_shape = input.shape[:-1] + (original_sample_size * num_values,)

    output = np.zeros(output_shape, dtype=dtype)

    flattened_data = input.reshape(-1)
    flattened_output = output.reshape((-1, num_values))

    for start in range(0, len(flattened_data), len(_ARANGE)):
        l = min(len(_ARANGE), len(flattened_data) - start)
        output_region = flattened_output[start : start + l]
        output_region[_ARANGE[:l], flattened_data[start : start + l]] = 1

    return output
