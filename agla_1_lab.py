import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    matrix = np.hstack((A, b.reshape(n, 1)))
    for i in range(n):
        max_row = np.argmax(np.abs(matrix[i:, i])) + i
        if matrix[max_row, i] == 0:
            raise ValueError("The matrix is singular or nearly singular.")
        matrix[[i, max_row]] = matrix[[max_row, i]]
        for j in range(i + 1, n):
            factor = matrix[j, i] / matrix[i, i]
            matrix[j] = matrix[j] - factor * matrix[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (matrix[i, -1] - np.dot(matrix[i, i + 1:n], x[i + 1:])) / matrix[i, i]
    return x

def matrix(size):
    print(f"Enter the coefficient matrix ({size}x{size}):")
    print("Enter each row with numbers separated by spaces.")
    A = []
    for i in range(size):
        while True:
            try:
                row = list(map(float, input(f"Row {i + 1}: ").split()))
                if len(row) != size:
                    raise ValueError(f"Expected {size} elements in row {i + 1}.")
                A.append(row)
                break
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
    return np.array(A, dtype=float)
def vector(size):
    print(f"Enter the right-hand side vector (size {size}):")
    print("Please enter the numbers separated by spaces.")
    while True:
        try:
            b = list(map(float, input().split()))
            if len(b) != size:
                raise ValueError(f"Expected {size} elements in the vector.")
            break
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
    return np.array(b, dtype=float)

if __name__ == "__main__":
    try:
        size = int(input("Enter the number of equations: "))
        A = matrix(size)
        b = vector(size)
        x = gaussian_elimination(A, b)
        print("Answer:")
        for i in range(size):
            print(f"x[{i + 1}] = {x[i]:.4f}")
    except ValueError as e:
        print(f"Error: {e}")
