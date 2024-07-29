from numpy import ones, real, zeros
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.sparse import csr_matrix, diags, vstack

class Difference_Gradient:    

    def __init__(self, dimensions, transpose=False):
        self.transpose = transpose
        self.no_of_rows, self.no_of_columns = dimensions
        ones_x = ones(self.no_of_columns)
        ones_y = ones(self.no_of_rows)        
        self.Dx = diags(diagonals=[-ones_x, ones_x], offsets=[0, 1], \
            shape=(self.no_of_columns-1, self.no_of_columns), format='csr')
        self.Dx = vstack([self.Dx, csr_matrix((1, self.no_of_columns))])
        self.Dy = diags(diagonals=[-ones_y, ones_y], offsets=[0, 1], \
            shape=(self.no_of_rows-1, self.no_of_rows), format='csr')
        self.Dy = vstack([self.Dy, csr_matrix((1, self.no_of_rows))])

    def __matmul__(self, argument):        
        if self.transpose is False:            
            output = zeros((self.no_of_rows, self.no_of_columns, 2))
            output[:, :, 0] = argument @ self.Dx.T
            output[:, :, 1] = self.Dy @ argument
        else:
            output = argument[:, :, 0] @ self.Dx + self.Dy.T @ argument[:, :, 1]
        return output

    @property
    def T(self):
        dimensions = [self.no_of_rows, self.no_of_columns]
        return Difference_Gradient(dimensions, transpose=not self.transpose)

class Fourier_Transform:

    def __init__(self, transpose=False):        
        self.transpose = transpose

    def __matmul__(self, argument):
        if self.transpose is False:
            return fftshift(fft2(argument, norm='ortho'))
        else:
            return real(ifft2(ifftshift(argument), norm='ortho'))

    @property
    def T(self):
        return Fourier_Transform(transpose=not self.transpose)

class Subsampling:

    def __init__(self, mask:bool, transpose=False):
        self.mask = mask
        self.transpose = transpose

    def __matmul__(self, argument):
        return self.mask * argument

    @property
    def shape(self):
        no_of_rows = prod(self.mask.shape)
        return [no_of_rows, no_of_rows]

    @property
    def T(self):
        return Subsampling(self.mask, transpose=not self.transpose)

class Fourier_Subsampling(Subsampling, Fourier_Transform):

    def __init__(self, mask:bool, transpose=False):
        Subsampling.__init__(self, mask, transpose)
        Fourier_Transform.__init__(self, transpose)

    def __matmul__(self, argument):
        if self.transpose is False:
            ft_image = Fourier_Transform.__matmul__(self, argument)
            return Subsampling.__matmul__(self, ft_image)
        else:
            ft_image = Subsampling.__matmul__(self, argument)
            return Fourier_Transform.__matmul__(self, ft_image)

    @property
    def T(self):
        return Fourier_Subsampling(self.mask, transpose=not self.transpose)