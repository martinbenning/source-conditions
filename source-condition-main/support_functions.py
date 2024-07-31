from numpy import asarray, concatenate, Inf, ones
from numpy.linalg import norm
from numpy.random import randn
from scipy.sparse import csr_matrix, diags, identity, kron, vstack

def fwht(x):
    """Recursive Fast Walsh-Hadamard Transform of array x."""
    x = asarray(x)
    n = x.shape[0]
    if n == 1:
        return x
    else:
        x_even = fwht(x[::2])
        x_odd = fwht(x[1::2])
        combined = concatenate([x_even + x_odd, x_even - x_odd])
        return combined

def ifwht(x):
    """Inverse Fast Walsh-Hadamard Transform of array x."""
    x = fwht(x)
    return x / x.shape[0]

def is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0

def norm_estimator(operator, tolerance=1e-3, max_no_of_iterations=150):    

    vector = randn(operator.shape[1], 1)    
    counter = 0
    sensitivity = Inf
    while (counter < max_no_of_iterations) and (sensitivity > tolerance):
        previous_vector = vector
        vector = operator.T @ (operator @ vector)
        vector_norm = norm(vector, 2)
        vector = vector / vector_norm
        counter += 1
        sensitivity = norm(vector - previous_vector, 2)/vector_norm
    print("Completed after {i} iterations, with tolerance {t}.".format( \
            i = counter, t = sensitivity))
    return norm(operator @ vector, 2)/norm(vector, 2)

def sparse_finite_difference_gradient(dimensions, zero_rows=False):

    no_of_rows, no_of_columns = dimensions
    ones_x = ones(no_of_columns)
    ones_y = ones(no_of_rows)
    eye_x = identity(no_of_columns, format='csr')
    eye_y = identity(no_of_rows, format='csr')
    Dx = diags(diagonals=[-ones_x, ones_x], offsets=[0, 1], \
            shape=(no_of_columns-1, no_of_columns), format='csr')
    if zero_rows:
        Dx = vstack([Dx, csr_matrix((1, no_of_columns))])
    Dy = diags(diagonals=[-ones_y, ones_y], offsets=[0, 1], \
            shape=(no_of_rows-1, no_of_rows), format='csr')
    if zero_rows:
        Dy = vstack([Dy, csr_matrix((1, no_of_rows))])
    DX = kron(Dx, eye_y)
    DY = kron(eye_x, Dy)
    return vstack([DX, DY])