from flask import Flask, render_template, request, jsonify
import sympy as sp

app = Flask(__name__)

def qr_decomposition(matrix):
    Q, R = matrix.QRdecomposition()
    return Q, R

@app.route('/', methods=['GET'])
def index():
    # Simply render the template on GET request
    return render_template('index.html')

@app.route('/decompose', methods=['POST'])
def decompose():
    # Retrieve matrix from AJAX POST data
    data = request.json
    matrix_input = data['matrix']
    # Convert input to sympy Matrix
    matrix = sp.Matrix(matrix_input)
    # Compute QR decomposition
    Q, R = qr_decomposition(matrix)
    # Generate LaTeX code for display
    Q_latex = sp.latex(Q)
    R_latex = sp.latex(R)
    # Respond with JSON
    return jsonify({'Q': Q_latex, 'R': R_latex})

if __name__ == '__main__':
    app.run(debug=True)
