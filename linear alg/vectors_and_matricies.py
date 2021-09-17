import numpy as np
from numpy.linalg import norm
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot = a.dot(b)

#Norm 1
n1 = norm(a, 1)

#Norm 2
#default one
n2 = norm(a, 2)
n2 = norm(a)

# create a 3 X 3 matrix
M = np.ones([3, 3]) *  np.array([1, 2, 3])

# Triangular matrix
lower_triangular_matrix = np.tril(M)
upper_triangular_matrix = np.triu(M)

# Diagonal matrix

# create a diagonal matrix
diagonal_matrix = np.diag(a)

# find out a diagonal from a given matrix
diagonal = np.diag(diagonal_matrix)

# identity matrix
i = np.identity(4)

# transpose
transpose = M.T

# inverse
M = np.random.randint(33, size=[4, 4])
m_inverse = np.linalg.inv(M)

# trace or diaonal sum
trace = np.trace(M)

# determinant
M = np.array([[1, 2],
	          [3, 4]])

det = np.linalg.det(M)

# rank
rank = np.linalg.matrix_rank(M)
print(rank)