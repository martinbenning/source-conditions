import numpy as np

def test_hadamard_transform():
    # Test case 1: 2x2 Hadamard transform
    dimensions = [2, 2]
    transform = Hadamard_Transform(dimensions)
    input_matrix = np.array([[1, 2], [3, 4]])
    expected_output = np.array([[5, -1], [1, -3]])
    output = transform @ input_matrix
    assert np.allclose(output, expected_output)

    # Test case 2: 4x4 Hadamard transform
    dimensions = [4, 4]
    transform = Hadamard_Transform(dimensions)
    input_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    expected_output = np.array([[34, -2, -2, 2], [-2, 2, -2, 2], [-2, -2, 2, 2], [-2, 2, 2, -2]])
    output = transform @ input_matrix
    assert np.allclose(output, expected_output)

    # Test case 3: 8x8 Hadamard transform
    dimensions = [8, 8]
    transform = Hadamard_Transform(dimensions)
    input_matrix = np.random.rand(8, 8)
    output = transform @ input_matrix
    reconstructed_input = transform.T @ output
    assert np.allclose(input_matrix, reconstructed_input)

test_hadamard_transform()