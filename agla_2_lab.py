import numpy as np

def gauss_jordan_inverse(A):
    n = len(A)
    Aug = np.hstack([A, np.eye(n)])

    for i in range(n):
        if Aug[i, i] == 0:
            raise ValueError("Impossible to determine the inverse matrix")
        Aug[i] = Aug[i] / Aug[i, i]
        for j in range(n):
            if i != j:
                Aug[j] -= Aug[j, i] * Aug[i]

    return Aug[:, n:]

n = int(input("Size of matrix: "))
A = np.array([list(map(float, input(f"Line {i+1}: ").split())) for i in range(n)])

try:
    A_inv = gauss_jordan_inverse(A)
    print("Result:")
    print(A_inv)
except ValueError as e:
    print(e)
