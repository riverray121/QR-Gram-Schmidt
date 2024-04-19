from sympy import Matrix, sqrt, latex
import sympy as sp

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def vector_subtract(v1, v2):
    return [x - y for x, y in zip(v1, v2)]

def scalar_multiply(c, v):
    return [c * x for x in v]

def norm(v):
    return sqrt(sum(x**2 for x in v))

def format_vector(v):
    return '\\begin{bmatrix}' + ' \\\\ '.join(latex(sp.nsimplify(x.evalf())) for x in v) + '\\end{bmatrix}'

def format_scalar(s):
    return latex(sp.nsimplify(s.evalf()))

def format_matrix(m):
    M = Matrix(m)
    M.applyfunc(sp.nsimplify)
    M = sp.latex(M)
    return M

def gram_schmidt(vectors):
    orthogonal = []
    steps_explanation = ["\\text{Gram-Schmidt Orthogonalization Process:}"]
    for i, v in enumerate(vectors):
        v_matrix = Matrix([v])
        steps_explanation.append(f"\\text{{Step {i+1}:}}")
        steps_explanation.append(f"\\text{{Start with }} v_{i+1} = {format_vector(v_matrix)}.")

        for j, u in enumerate(orthogonal):
            proj_factor = dot_product(v, u) / dot_product(u, u)
            proj = scalar_multiply(proj_factor, u)
            proj_matrix = Matrix([proj])
            steps_explanation.append(
                f"\\text{{Project }} v_{i+1} \\text{{ onto }} u_{j+1} \\text{{ gives }} \\text{{proj}}_{i+1}^{j+1} = "
                f"{format_scalar(proj_factor)} \\times {format_vector(u)} = {format_vector(proj)}."
            )
            v = vector_subtract(v, proj)

        v_norm = norm(v)
        if v_norm != 0:
            v = scalar_multiply(1 / v_norm, v)
        orthogonal.append(v)
        v_matrix = Matrix([v])
        steps_explanation.append(
            f"\\text{{After orthogonalization, }} u_{i+1} = {format_vector(v)}."
        )
        if v_norm != 0:
            steps_explanation.append(
                f"\\text{{Normalize }} u_{i+1} \\text{{ to get the orthonormal vector }} q_{i+1} = "
                f"\\frac{{1}}{{{format_scalar(v_norm)}}} \\times {format_vector(v)} = {format_vector(v)}."
            )
        else:
            steps_explanation.append(
                f"\\text{{Vector }} v_{i+1} \\text{{ is zero after orthogonalization (it's dependent), so we skip normalization.}}"
            )

    return [[float(v) for v in vec] for vec in orthogonal], steps_explanation

def qr_decomposition(A):
    A = Matrix(A)
    num_rows, num_cols = A.shape
    Q, R = A.QRdecomposition()
    
    # Generate explanations
    qr_steps_explanation = [
        "\\text{QR Decomposition Process:}",
        "\\text{The goal is to decompose the matrix A into an orthogonal matrix Q and an upper triangular matrix R.}",
        "\\text{This is achieved through the Gram-Schmidt process.}"
    ]
    
    # Start the Gram-Schmidt process
    for j in range(num_cols):
        a_col = A[:, j]
        qr_steps_explanation.append(f"\\text{{Consider the column vector {j+1} of A:}} {format_vector(a_col)}")

        if j == 0:
            qr_steps_explanation.append("\\text{Since this is the first vector, it's already orthogonal to the preceding ones (as there are none).}")
        else:
            qr_steps_explanation.append("\\text{We orthogonalize this column with respect to the previous columns of Q.}")

        for i in range(j):
            r_ij = R[i, j]
            q_i = Q[:, i]
            qr_steps_explanation.append(
                f"\\begin{{gather*}}\\text{{The projection of column {j+1} onto the normalized column {i+1} of Q is given by the inner product of A's column with Q's column,}}\\\\\\text{{scaled by the corresponding entry in R.}}\\end{{gather*}}"
            )
            qr_steps_explanation.append(
                f"r_{{ {i+1}, {j+1} }} = \\langle {format_vector(a_col)}, {format_vector(q_i)} \\rangle = {format_scalar(r_ij)}"
            )

        q_j = Q[:, j]
        r_jj = R[j, j]
        qr_steps_explanation.append(
            f"\\text{{After orthogonalizing, we normalize the result to get the column {j+1} of Q:}} {format_vector(q_j)}"
        )
        qr_steps_explanation.append(
            f"\\text{{The normalization factor is the diagonal entry of R, which is the norm of the column before normalization: }} r_{{ {j+1}, {j+1} }} = {format_scalar(r_jj)}"
        )

    # Explain how the matrices Q and R are formed
    qr_steps_explanation.append(
        "\\text{Now that we have orthogonalized all columns of A, we compile them to form the matrix Q:}"
    )
    qr_steps_explanation.append(f"Q = {format_matrix(Q)}")
    qr_steps_explanation.append(
        "\\text{The upper triangular matrix R is formed from the coefficients used during the orthogonalization:}"
    )
    qr_steps_explanation.append(f"R = {format_matrix(R)}")
    qr_steps_explanation.append(
        "\\text{The product of Q and R will give us the original matrix A, verifying the decomposition.}"
    )

    return Q.evalf().tolist(), R.evalf().tolist(), qr_steps_explanation

def transpose(matrix):
    return list(map(list, zip(*matrix)))
