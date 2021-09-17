import numpy as np

from scipy.sparse import csr_matrix
import scipy as sp
A = np.array([[1, 0, 0, 1, 0, 0],
			  [0, 0, 2, 0, 0, 1],
			  [0, 0, 0, 2, 0, 0]])
print('-'*10)
print(A)

S =  csr_matrix(A)
print('-'*10)
print(S)

B = S.todense()
print('-'*10)
print(B)
print('-'*10)

sparsity = 1.0 - np.count_nonzero(A)/A.size
print(sparsity)