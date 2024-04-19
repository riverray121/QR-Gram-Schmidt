from sympy import Matrix, sqrt, latex, zeros
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
    # Assume vectors are the columns of the matrix input
    orthogonal = []
    steps_explanation = ["\\text{Gram-Schmidt Orthogonalization Process:}"]
    
    A = Matrix(vectors)
    if A.rank() < A.cols:
        steps_explanation.append("\\text{The input matrix is not linearly independent, which is a requirement for our Gram-Schmidt Calculator.}")
        return [], steps_explanation
    
    for i in range(len(vectors[0])):  # Iterate through each column index
        # Extract the ith column as a vector
        v = [row[i] for row in vectors]
        v_matrix = Matrix(v)
        steps_explanation.append(f"\\text{{Step {i+1}:}}")
        steps_explanation.append(f"\\text{{Start with }} v_{i+1} = {format_vector(v_matrix)}.")

        for j, u in enumerate(orthogonal):
            u_matrix = Matrix(u)
            proj_factor = dot_product(v, u) / dot_product(u, u)
            proj = scalar_multiply(proj_factor, u)
            proj_matrix = Matrix(proj)
            steps_explanation.append(
                f"\\text{{Project }} v_{i+1} \\text{{ onto }} u_{j+1} \\text{{ gives }} "
                f"\\text{{proj}}_{i+1}^{j+1} = {format_scalar(proj_factor)} \\times {format_vector(u_matrix)} = {format_vector(proj_matrix)}."
            )
            v = vector_subtract(v, proj)

        v_norm = norm(v)
        if v_norm != 0:
            v = scalar_multiply(1 / v_norm, v)
        orthogonal.append(v)

        v_matrix_normalized = Matrix(v)
        steps_explanation.append(
            f"\\text{{After orthogonalization, }} u_{i+1} = {format_vector(v_matrix_normalized)}."
        )
        if v_norm != 0:
            steps_explanation.append(
                f"\\text{{Normalize }} u_{i+1} \\text{{ to get the orthonormal vector }} q_{i+1} = "
                f"\\frac{{1}}{{{format_scalar(v_norm)}}} \\times {format_vector(v_matrix_normalized)} = {format_vector(v_matrix_normalized)}."
            )
        else:
            steps_explanation.append(
                f"\\text{{Vector }} v_{i+1} \\text{{ is zero after orthogonalization (it's dependent), so we skip normalization.}}"
            )

    # Transpose the orthogonal matrix to return it to the original orientation
    orthogonal_transposed = transpose(orthogonal)
    return [[float(v) for v in vec] for vec in orthogonal_transposed], steps_explanation


def qr_decomposition(A):
    A = Matrix(A)
    num_rows, num_cols = A.shape
    Q = zeros(num_rows, num_cols)
    R = zeros(num_cols, num_cols)
    
    qr_steps_explanation = [
        "\\text{QR Decomposition Process:}",
        "\\text{The goal is to decompose the matrix A into an orthogonal matrix Q and an upper triangular matrix R.}",
        "\\text{This is achieved through the Gram-Schmidt process.}"
    ]
    
    if A.rank() < A.cols:
        qr_steps_explanation = ["\\text{The input matrix is not linearly independent, which is a requirement for our QR decomposition Calculator.}"]
        print(qr_steps_explanation)
        return [], [], qr_steps_explanation
    
    for j in range(num_cols):
        a_col = A[:, j]
        
        if a_col.norm() == 0:
            qr_steps_explanation.append(f"\\text{{Column {j+1} of A is initially zero, thus directly setting Q and R columns for {j+1} to zero.}}")
            Q[:, j] = zeros(num_rows, 1)
            R[j, j] = 0
            continue
        
        qr_steps_explanation.append(f"\\text{{Consider the column vector {j+1} of A:}} {format_vector(a_col)}")
        
        for i in range(j):
            r_ij = Q[:, i].dot(a_col)
            R[i, j] = r_ij
            a_col -= Q[:, i] * r_ij
            qr_steps_explanation.append(
                f"\\text{{The projection of column {j+1} onto the normalized column {i+1} of Q is given by the inner product of A's column with Q's column,}}"
            )
            qr_steps_explanation.append(
                f"\\text{{scaled by the corresponding entry in R.}}"
            )
            qr_steps_explanation.append(
                f"r_{{ {i+1}, {j+1} }} = \\langle {format_vector(a_col)}, {format_vector(Q[:, i])} \\rangle = {format_scalar(r_ij)}"
            )
        
        r_jj = a_col.norm()
        if r_jj == 0:
            qr_steps_explanation.append("\\text{Column is linearly dependent or zero after projection; normalization skipped.}")
            Q[:, j] = zeros(num_rows, 1)
            R[j, j] = 0
        else:
            Q[:, j] = a_col / r_jj
            R[j, j] = r_jj
            qr_steps_explanation.append(
                f"\\text{{After orthogonalizing, we normalize the result to get the column {j+1} of Q:}} {format_vector(Q[:, j])}"
            )
            qr_steps_explanation.append(
                f"\\text{{The normalization factor is the diagonal entry of R, which is the norm of the column before normalization: }} r_{{ {j+1}, {j+1} }} = {format_scalar(r_jj)}"
            )
    
    qr_steps_explanation.append("\\text{Now that we have orthogonalized all columns of A, we compile them to form the matrix Q:}")
    qr_steps_explanation.append(f"Q = {format_matrix(Q)}")
    qr_steps_explanation.append("\\text{The upper triangular matrix R is formed from the coefficients used during the orthogonalization:}")
    qr_steps_explanation.append(f"R = {format_matrix(R)}")
    qr_steps_explanation.append("\\text{The product of Q and R will give us the original matrix A, verifying the decomposition.}")

    return Q.evalf(chop=True).tolist(), R.evalf(chop=True).tolist(), qr_steps_explanation

def transpose(matrix):
    return list(map(list, zip(*matrix)))