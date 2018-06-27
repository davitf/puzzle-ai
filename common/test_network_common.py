import numpy as np

from common import network_common


def test_onehot_encode():
    sample_size = 3
    num_features = 5
    num_samples = 2
    data = (
        np.arange(num_samples * sample_size).reshape((num_samples, sample_size))
        % num_features
    )

    encoded = network_common.onehot_encode(data, num_features)
    assert encoded.shape == (num_samples, sample_size * num_features)

    # fmt: off
    expected_output = [
        [1,0,0,0,0,  0,1,0,0,0,  0,0,1,0,0,],   # data = [[0, 1, 2],
        [0,0,0,1,0,  0,0,0,0,1,  1,0,0,0,0,],   #         [3, 4, 0]]
    ]
    # fmt: on

    assert encoded.tolist() == expected_output


def test_onehot_encode_with_dtype():
    sample_size = 3
    num_features = 5
    num_samples = 2
    data = (
        np.arange(num_samples * sample_size).reshape((num_samples, sample_size))
        % num_features
    )

    encoded = network_common.onehot_encode(data, num_features, dtype="int")
    assert encoded.shape == (num_samples, sample_size * num_features)
    assert encoded.dtype == np.int

    # fmt: off
    expected_output = [
        [1,0,0,0,0,  0,1,0,0,0,  0,0,1,0,0,],   # data = [[0, 1, 2],
        [0,0,0,1,0,  0,0,0,0,1,  1,0,0,0,0,],   #         [3, 4, 0]]
    ]
    # fmt: on

    assert encoded.tolist() == expected_output


def test_onehot_encode_newaxis():
    sample_size = 3
    num_features = 5
    num_samples = 2
    data = (
        np.arange(num_samples * sample_size).reshape((num_samples, sample_size))
        % num_features
    )

    encoded = network_common.onehot_encode(data, num_features, new_axis=True)
    assert encoded.shape == (num_samples, sample_size, num_features)

    # fmt: off
    expected_output = [
        [[1,0,0,0,0,], [0,1,0,0,0,], [0,0,1,0,0,]],   # data = [[0, 1, 2],
        [[0,0,0,1,0,], [0,0,0,0,1,], [1,0,0,0,0,]] ,  #         [3, 4, 0]]
    ]
    # fmt: on

    assert encoded.tolist() == expected_output


def test_onehot_encode_with_multidimensional_large_data():
    # This should be large enough to ensure that onehot_encode will process
    # the data in multiple chunks.
    num_features = 16
    input_shape = (10000, 7, 11)
    input = np.random.randint(0, num_features, input_shape)

    encoded = network_common.onehot_encode(input, num_features, new_axis=True)

    # Decode the encoded data by finding the index of the 1 for each value.
    decoded = np.argmax(encoded, axis=-1)

    assert input.shape == decoded.shape
    assert input.tolist() == decoded.tolist()
