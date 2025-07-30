"""
Re-invent functions for attentions. Why not?
"""
import math

import numpy as np


def softmax(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of a matrix along the last axis.
    
    Args:
        matrix (2D np.ndarray): Input matrix.
        
    Returns:
        2D np.ndarray: Softmax of the input matrix.

    NOTE: scipy.special.softmax does same thing.
    """
    assert matrix.ndim == 2, "Input matrix must be 2-dimensional."

    matrix_exp = np.exp(matrix)
    row_sum = matrix_exp.sum(axis=-1, keepdims=True)
    return matrix_exp / row_sum


def attention(
    query_matrix: np.ndarray,
    key_matrix: np.ndarray,
    value_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute the attention output given query, key, and value matrices.
    
    Args:
        query_matrix (2D np.ndarray): Query matrix.
        key_matrix (2D np.ndarray): Key matrix.
        value_matrix (2D np.ndarray): Value matrix.
        
    Returns:
        2D np.ndarray: Attention output.
    """
    assert query_matrix.ndim == 2, "Query matrix must be 2-dimensional."
    assert key_matrix.ndim == 2, "Key matrix must be 2-dimensional."
    assert value_matrix.ndim == 2, "Value matrix must be 2-dimensional."

    scores = query_matrix @ key_matrix.T
    attention_weights = softmax(scores) / math.sqrt(key_matrix.shape[-1])
    return attention_weights @ value_matrix