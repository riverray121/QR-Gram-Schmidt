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
def de