import numpy as np
import math


class SelfAttention():
    def __init__(self, d_model=8) -> None:
        self.d_model = d_model  # Vector's size

    @staticmethod
    def dot_product(matrix_1, matrix_2):
        return np.matmul(matrix_1, matrix_2)
    
    def scale(self, input_matrix):
        return input_matrix / (math.sqrt(self.d_model))

    def softmax(self, input_matrix):
        input_matrix_exp = np.exp(input_matrix)
        return input_matrix_exp / np.sum(input_matrix_exp, axis=-1, keepdims=True)

    def call(self, seq_len=4):
        q = np.random.randn(seq_len, self.d_model)
        k = np.random.randn(seq_len, self.d_model)
        v = np.random.randn(seq_len, self.d_model)

        result = self.dot_product(q, k.T)
        result = self.scale(result)
        result = self.softmax(result)
        result = self.dot_product(result, v)
        return result


block = SelfAttention(d_model=10)
print(block.forward(5))
