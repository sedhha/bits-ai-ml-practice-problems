import numpy as np

# X = np.array([
#     [1.11, 1.21, 1.36, 1.49, 1.63, 1.68, 1.83, 1.88, 1.95],
#     [10, 12, 13, 15, 16, 17, 18, 19, 20]
# ])

# # U, Sigma, VT = np.linalg.svd(X, full_matrices=False)


# XTX = np.dot(X.T, X)
# eigenvalues, eigenvectors = np.linalg.eig(XTX)

# print(eigenvalues)

X = np.array([
    [0.07949647, 0.88825864],
    [0.88825864, 10.02469136 ]
])

eV, eVal = np.linalg.eig(X)

print(eV)
print(eVal)

U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
print(U)
