import numpy as np
from scipy.linalg import lu, svd

T = np.arange(1, 28).reshape((3, 3, 3))
print('*'*70)
#print(T)
print('*'*70)
#print(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))

#print(np.tensordot(T, np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])))



A = np.array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]])

# lu decomposition
p, l, u = lu(A)

# QR decomposition
q, r = np.linalg.qr(A)


# cholskey decomposition
A =  np.array([[2, 1, 1],
			   [1, 2, 1],
			   [1, 1, 2]
			  ])

# factorize
L = np.linalg.cholesky(A)

# reconstruct
B = L.dot(L.T)


# eigen decomposition
eig_val, eig_vect = np.linalg.eig(A)

# check eigen
def check_eigen():
	for val, each in zip(eig_val,eig_vect.T):
		print(A.dot(each))
		print(np.dot(val, each))
		print('*'*70)

def reconstruct_eigen_vectors(eig_val, eig_vect):
	Q = eig_vect
	R = np.linalg.inv(Q)
	L = np.diag(eig_val)

	return(Q.dot(L).dot(R))


# svd
A = np.array([
			[1, 2],
			[3, 4],
			[5, 6]])

U, S, V = svd(A)


def reconstruct_svd_vectors(U, s, V):
	sigma = np.zeros((A.shape[0], A.shape[1]))
	if A.shape[0] <= A.shape[1]:
		x = A.shape[0]
	else:
		x = A.shape[1]
	sigma[:x, :x] = np.diag(s)
	return U.dot(sigma).dot(V)


# pseudoinverse
# pseudoinverse
A_plus = np.linalg.pinv(A)
A_plus


# application of svd
# define matrix
A = np.array([
[1,2,3,4,5,6,7,8,9,10],
[11,12,13,14,15,16,17,18,19,20],
[21,22,23,24,25,26,27,28,29,30]])
print(A)

# factorize
U, s, V = svd(A)

# create m x n Sigma matrix
Sigma = np.zeros((A.shape[0], A.shape[1]))

# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)

# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
V = V[:n_elements, :]

# reconstruct
B = U.dot(Sigma.dot(V))
print('-'*10)
print(Sigma)
print('#'*10)
print('-'*10)
print(V)
print('#'*10)
# transform
T = U.dot(Sigma)
print(T)
T = A.dot(V.T)
print(T)