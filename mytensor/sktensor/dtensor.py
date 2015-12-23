import numpy as np
from numpy import array, prod, argsort
from .core import tensor_mixin
from .pyutils import inherit_docstring_from, from_to_without

__all__ = [
    'dtensor',
    'unfolded_dtensor',
]

class dtensor(tensor_mixin, np.ndarray):
    """
    Class to store **dense** tensors

    Parameters
    ----------
    input_array : np.ndarray
        Multidimenional numpy array which holds the entries of the tensor

    Examples
    --------
    Create dense tensor from numpy array

    >>> T = np.zeros((3, 4, 2))
    >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
    >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
    >>> T = dtensor(T)
    """

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def _ttm_compute(self, V, mode, transp):
        sz = array(self.shape)
        r1, r2 = from_to_without(0, self.ndim, mode, separate=True)
        order = [mode] + r1 + r2
        newT = self.transpose(axes=order)
        newT = newT.reshape(sz[mode], prod(sz[r1 + list(range(mode + 1, len(sz)))]))
        if transp:
            newT = V.T.dot(newT)
            p = V.shape[1]
        else:
            newT = V.dot(newT)
            p = V.shape[0]
        newsz = [p] + list(sz[:mode]) + list(sz[mode + 1:])
        newT = newT.reshape(newsz)
        newT = newT.transpose(argsort(order))
        return dtensor(newT)

    def unfold(self, mode):
        """
        Unfolds a dense tensor in mode n.

        Parameters
        ----------
        mode : int
            Mode in which tensor is unfolded

        Returns
        -------
        unfolded_dtensor : unfolded_dtensor object
            Tensor unfolded along mode

        Examples
        --------
        Create dense tensor from numpy array

        >>> T = np.zeros((3, 4, 2))
        >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
        >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
        >>> T = dtensor(T)

        Unfolding of dense tensors

        >>> T.unfold(0)
        array([[  1.,   4.,   7.,  10.,  13.,  16.,  19.,  22.],
               [  2.,   5.,   8.,  11.,  14.,  17.,  20.,  23.],
               [  3.,   6.,   9.,  12.,  15.,  18.,  21.,  24.]])
        >>> T.unfold(1)
        array([[  1.,   2.,   3.,  13.,  14.,  15.],
               [  4.,   5.,   6.,  16.,  17.,  18.],
               [  7.,   8.,   9.,  19.,  20.,  21.],
               [ 10.,  11.,  12.,  22.,  23.,  24.]])
        >>> T.unfold(2)
        array([[  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.,
                 12.],
               [ 13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,  22.,  23.,
                 24.]])
        """

        sz = array(self.shape)
        N = len(sz)
        order = ([mode], from_to_without(N - 1, -1, mode, step=-1, skip=-1))
        newsz = (sz[order[0]][0], prod(sz[order[1]]))
        arr = self.transpose(axes=(order[0] + order[1]))
        arr = arr.reshape(newsz)
        return unfolded_dtensor(arr, mode, self.shape)

    def norm(self):
        """
        Computes the Frobenius norm for dense tensors
        :math:`norm(X) = \sqrt{\sum_{i_1,\ldots,i_N} x_{i_1,\ldots,i_N}^2}`

        References
        ----------
        [Kolda and Bader, 2009; p.457]
        """
        return np.linalg.norm(self)

    @inherit_docstring_from(tensor_mixin)
    def transpose(self, axes=None):
        return dtensor(np.transpose(array(self), axes=axes))

class unfolded_dtensor(np.ndarray):

    def __new__(cls, input_array, mode, ten_shape):
        obj = np.asarray(input_array).view(cls)
        obj.ten_shape = ten_shape
        obj.mode = mode
        return obj
