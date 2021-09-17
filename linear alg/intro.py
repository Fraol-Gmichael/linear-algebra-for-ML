import numpy as np

# empty zeros and ones
empty = np.empty([3, 3])
zeros = np.zeros([3, 3])
ones = np.ones([3, 3])

# combining arrays vstack
a1 = np.array([1, 2, 3])
a2 = np.array([[4, 5, 6],
			   [7, 8, 9],
			   [10, 11, 12]])

a3 = np.vstack((a1, a2))
a4 = np.array([21, 4, 12, 14, 25])
a5 = np.hstack((a1, a4))

a = np.arange(16).reshape([4, 4])

'''print(a)
print()
print(a[:])
print()
print(a[:, 1])
print()
print(a[1:2, :])
'''
data = a
X, y = data[:, :-1], data[:, -1]


def add_column(features, target):
	target = target.reshape([target.shape[0], 1])
	return np.hstack([features, target])

print(add_column(X, y))

