from flask import Flask, render_template, request, jsonify
from sympy import Matrix
from qr_all_functions import qr_decomposition, gram_schmidt
import sympy as sp

app = Flask(__name__)

def safe_matrix_conversion(data):
    """Ensure all sub-lists are of equal length before conversion to a Matrix."""
    try:
        if data and isinstance(data[0], list):
            max_len = max(len(sublist) for sublist in data)
            # Extend each sublist with zeroes to the length of the longest sublist
            uniform_data = [sublist + [0] * (max_len - len(sublist)) for sublist in data]
            return uniform_data
        return data
    except (TypeError, ValueError):
        raise ValueError("Invalid matrix data provided.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/decompose/qr', methods=['POST'])
def decompose_qr():
    try:
        data = request.get_json()
        matrix_input = safe_matrix_conversion(data['matrix'])
        
        Q, R, explanations = qr_decomposition(matrix_input)

        # Convert Q and R to lists of lists of standard Python floats
        Q = [[sp.nsimplify(num) for num in row] for row in Q]
        R = [[sp.nsimplify(num) for num in row] for row in R]

        # Convert the matricies to latex
        Q = Matrix(Q)
        R = Matrix(R)
        Q = sp.latex(Q)
        R = sp.latex(R)

        return jsonify({
            'qr': {
                'Q': Q,
                'R': R,
            },
            'intermediary_steps': {
                'explanations': explanations
            }
        }), 200  # HTTP 200 OK
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400  # HTTP 400 Bad Request

@app.route('/decompose/gs', methods=['POST'])
def decompose_gs():
    try:
        data = request.get_json()
        matrix_input = safe_matrix_conversion(data['matrix'])
        
        orthogonal_vectors, explanations = gram_schmidt(matrix_input)

        # Convert orthogonal_vectors to lists of lists of standard Python floats
        orthogonal_vectors = [[sp.nsimplify(num) for num in row] for row in orthogonal_vectors]

        # Convert the Orthagonal vectors to latex 
        orthogonal_vectors = Matrix(orthogonal_vectors)
        orthogonal_vectors = sp.latex(orthogonal_vectors)

        return jsonify({
            'gs': {
                'orthogonal_vectors': orthogonal_vectors,
            },
            'intermediary_steps': {
                'explanations': explanations
            }
        }), 200  # HTTP 200 OK
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # HTTP 400 Bad Request

if __name__ == '__main__':
    app.run(debug=True)