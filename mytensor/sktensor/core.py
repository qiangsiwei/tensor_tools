import numpy as np
from numpy import array, dot, zeros, ones, arange
from numpy import setdiff1d
from scipy.linalg import eigh
from scipy.sparse import issparse as issparse_mat
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from abc import ABCMeta, abstractmethod
from .pyutils import is_sequence, func_attr

import sys
import types

class tensor_mixin(object):
    """
    Base tensor class from which all tensor classes are subclasses.
    """
    
    __metaclass__ = ABCMeta

    def ttm(self, V, mode=None, transp=False, without=False):
        """
        Tensor times matrix product

        Parameters
        ----------
        V : M x N array_like or list of M_i x N_i array_likes
            Matrix or list of matrices for which the tensor times matrix
            products should be performed
        mode : int or list of int's, optional
            Modes along which the tensor times matrix products should be
            performed
        transp: boolean, optional
            If True, tensor times matrix products are computed with
            transpositions of matrices
        without: boolean, optional
            It True, tensor times matrix products are performed along all
            modes **except** the modes specified via parameter ``mode``


        Examples
        --------
        Create dense tensor

        >>> T = zeros((3, 4, 2))
        >>> T[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
        >>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
        >>> T = dtensor(T)

        Create matrix

        >>> V = array([[1, 3, 5], [2, 4, 6]])

        Multiply tensor with matrix along mode 0

        >>> Y = T.ttm(V, 0)
        >>> Y[:, :, 0]
        array([[  22.,   49.,   76.,  103.],
            [  28.,   64.,  100.,  136.]])
        >>> Y[:, :, 1]
        array([[ 130.,  157.,  184.,  211.],
            [ 172.,  208.,  244.,  280.]])

        """
        if mode is None:
            mode = range(self.ndim)
        if isinstance(V, np.ndarray):
            Y = self._ttm_compute(V, mode, transp)
        elif is_sequence(V):
            dims, vidx = check_multiplication_dims(mode, self.ndim, len(V), vidx=True, without=without)
            Y = self._ttm_compute(V[vidx[0]], dims[0], transp)
            for i in xrange(1, len(dims)):
                Y = Y._ttm_compute(V[vidx[i]], dims[i], transp)
        return Y

    @abstractmethod
    def transpose(self, axes=None):
        """
        Compute transpose of tensors.

        Parameters
        ----------
        axes : array_like of ints, optional
            Permute the axes according to the values given.

        Returns
        -------
        d : tensor_mixin
            tensor with axes permuted.

        See also
        --------
        dtensor.transpose, sptensor.transpose
        """
        pass

def istensor(X):
    return isinstance(X, tensor_mixin)

# dynamically create module level functions
conv_funcs = [
    'norm',
    'transpose',
    'ttm',
    'ttv',
    'unfold',
]

for fname in conv_funcs:
    def call_on_me(obj, *args, **kwargs):
        if not istensor(obj):
            raise ValueError('%s() object must be tensor (%s)' % (fname, type(obj)))
        func = getattr(obj, fname)
        return func(*args, **kwargs)

    nfunc = types.FunctionType(
        func_attr(call_on_me, 'code'),
        {
            'getattr': getattr,
            'fname': fname,
            'istensor': istensor,
            'ValueError': ValueError,
            'type': type
        },
        name=fname,
        argdefs=func_attr(call_on_me, 'defaults'),
        closure=func_attr(call_on_me, 'closure')
    )
    setattr(sys.modules[__name__], fname, nfunc)


def check_multiplication_dims(dims, N, M, vidx=False, without=False):
    dims = array(dims, ndmin=1)
    if len(dims) == 0:
        dims = arange(N)
    if without:
        dims = setdiff1d(range(N), dims)
    if not np.in1d(dims, arange(N)).all():
        raise ValueError('Invalid dimensions')
    P = len(dims)
    sidx = np.argsort(dims)
    sdims = dims[sidx]
    if vidx:
        if M > N:
            raise ValueError('More multiplicants than dimensions')
        if M != N and M != P:
            raise ValueError('Invalid number of multiplicants')
        if P == M:
            vidx = sidx
        else:
            vidx = sdims
        return sdims, vidx
    else:
        return sdims

def nvecs(X, n, rank, do_flipsign=True, dtype=np.float):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    Xn = X.unfold(n)
    if issparse_mat(Xn):
        Xn = csr_matrix(Xn, dtype=dtype)
        Y = Xn.dot(Xn.T)
        _, U = eigsh(Y, rank, which='LM')
    else:
        Y = Xn.dot(Xn.T)
        N = Y.shape[0]
        _, U = eigh(Y, eigvals=(N - rank, N - 1))
    # reverse order of eigenvectors such that eigenvalues are decreasing
    U = array(U[:, ::-1])
    # flip sign
    if do_flipsign:
        U = flipsign(U)
    return U

def flipsign(U):
    """
    Flip sign of factor matrices such that largest magnitude
    element will be positive
    """
    midx = abs(U).argmax(axis=0)
    for i in range(U.shape[1]):
        if U[midx[i], i] < 0:
            U[:, i] = -U[:, i]
    return U
