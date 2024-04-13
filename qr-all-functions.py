def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def vector_subtract(v1, v2):
    return [x - y for x, y in zip(v1, v2)]

def scalar_multiply(c, v):
    return [c * x for x in v]

def norm(v):
    return sum(x**2 for x in v)**0.5

def gram_schmidt(vectors):
    """Apply Gram-Schmidt process to a list of vectors to produce orthogonal vectors."""
    orthogonal = []
    for v in vectors:
        for u in orthogonal:
            proj = scalar_multiply(dot_product(v, u) / dot_product(u, u), u)
            v = vector_subtract(v, proj)
        orthogonal.append(v)
    return orthogonal

def qr_decomposition(A):
    """Perform QR decomposition using the Gram-Schmidt process."""
    num_rows, num_cols = len(A), len(A[0])
    transposed_A = transpose(A)
    Q_transposed = gram_schmidt(transposed_A)
    Q = transpose(Q_transposed)
    R = [[0] * num_cols for _ in range(num_cols)]

    for i in range(num_cols):
        for j in range(i, num_cols):
            if i == j:
                R[i][j] = norm(Q_transposed[i])
            else:
                R[i][j] = dot_product(Q_transposed[i], transposed_A[j]) / norm(Q_transposed[i])
            Q_transposed[i] = scalar_multiply(1 / norm(Q_transposed[i]), Q_transposed[i])

    Q = transpose(Q_transposed)
    return Q, R

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

# Example usage
A = [
    [12, -51, 4],
    [6, 167, -68],
    [-4, 24, -41]
]

Q, R = qr_decomposition(A)

# Displaying the results
print("Q Matrix:")
for row in Q:
    print(row)
print("\nR Matrix:")
for row in R:
    print(row)
