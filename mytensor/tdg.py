# -*- coding: UTF-8 -*-

import numpy as np
from sktensor import dtensor, tucker_hooi


def Gradient_Tensor_Decomposition(A, core_dim, lambda1=0.0001, epsilon=0.01):
	# Initialization
	len_X, len_Y, len_Z = A.shape
	C = dtensor(np.random.uniform(0, 5, size=(core_dim, core_dim, core_dim)))
	X = np.random.uniform(0, 5, size=(len_X, core_dim))
	Y = np.random.uniform(0, 5, size=(len_Y, core_dim))
	Z = np.random.uniform(0, 5, size=(len_Z, core_dim))

	# Iteration
	T = C.ttm(X, 0).ttm(Y, 1).ttm(Z, 2)
	loss_last, loss_curr, step = float('inf'), (T-A).std(), 0
	while loss_last - loss_curr > epsilon:
		loss_last = loss_curr
		for dim1 in xrange(len_X):
			for dim2 in xrange(len_Y):
				for dim3 in xrange(len_Z):
					if A[dim1][dim2][dim3] != 0:
						a = A[dim1][dim2][dim3]
						y = C.ttm(np.array([X[dim1]]), 0)\
							 .ttm(np.array([Y[dim2]]), 1)\
							 .ttm(np.array([Z[dim3]]), 2)[0][0][0]
						Xi, Yi, Zi = np.array([X[dim1]]), np.array([Y[dim2]]), np.array([Z[dim3]])
						step += 1
						nita = 1./step**0.5
						X[dim1] = Xi-nita*(y-a)*C.ttm(Yi, 1).ttm(Zi, 2).reshape((1,core_dim))-nita*lambda1*Xi
						Y[dim2] = Yi-step*(y-a)*C.ttm(Xi, 0).ttm(Zi, 2).reshape((1,core_dim))-step*lambda1*Yi
						Z[dim3] = Zi-step*(y-a)*C.ttm(Xi, 0).ttm(Yi, 1).reshape((1,core_dim))-step*lambda1*Zi
						C = C-step*(y-a)*np.kron(np.kron(Xi,Yi),Zi).reshape((core_dim, core_dim, core_dim))-step*lambda1*C
						T = C.ttm(X, 0).ttm(Y, 1).ttm(Z, 2)
						loss_curr = (T-A).std()
						print "loss_curr", loss_curr
	return C, [X, Y, Z]


if __name__ == "__main__":
	# Input
	A = np.zeros((3, 4, 2))
	A[:, :, 0] = [[ 1,  4,  7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
	A[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
	Gradient_Tensor_Decomposition(A, 2, 0.0001, 0)

