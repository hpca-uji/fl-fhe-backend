from encryption.factory import HomomorphicEncrytionFactory
import numpy as np
from core.util.common import read_file, write_file
import tomli
import os
import tenseal as ts

def main():
    he_backend = HomomorphicEncrytionFactory.get_backend(library='TENSEAL', schema='CKKS')
    he_backend.set_poly_modulus_degree(8192)
    he_backend.generate_keys()
    
    vector = np.array([1.5, 2.6, 3.8])
    encrypted_vector = he_backend.encrypt(vector)
    dencrypted_vector = he_backend.decrypt(encrypted_vector)
    print(dencrypted_vector)

    matrix = np.matrix([[1.1, 2.2], [3.3, 4.4]])
    encrypted_matrix = he_backend.encrypt(matrix)
    dencrypted_matrix = he_backend.decrypt(encrypted_matrix)
    print(dencrypted_matrix)
    
    
    vector1 = np.array([1.5, 2.6, 3.8])
    vector2 = np.array([1.5, 2.6, 3.8])
    
    encrypted_vector1 = he_backend.encrypt(vector1)
    encrypted_vector2 = he_backend.encrypt(vector2)
    dot_product = he_backend.enc_dot(encrypted_vector1, encrypted_vector2)
    dencrypted_vector = he_backend.decrypt(dot_product)
    
    matrix1 = np.matrix([[1.1, 2.2], [3.3, 4.4]])
    matrix2 = np.matrix([[1.1, 2.2], [3.3, 4.4]])
    
    encrypted_matrix1 = he_backend.encrypt(matrix1)
    encrypted_matrix2 = he_backend.encrypt(matrix2)
    dot_matrix = he_backend.enc_dot(encrypted_matrix1, encrypted_matrix2)
    dencrypted_dot_matrix = he_backend.decrypt(dot_matrix)
    print(dencrypted_dot_matrix)
    
    
    matrix = np.matrix([[1.1, 2.2], [3.3, 4.4]])
    vector = np.array([1.5, 2.6])
    
    plain_matrix = he_backend.plain_tensor(matrix)
    encrypted_vector = he_backend.encrypt(vector)
    matmul = he_backend.enc_matmul(encrypted_vector, plain_matrix)
    dencrypted_matmul = he_backend.decrypt(matmul)
    print(dencrypted_matmul)
    
    
if __name__ == '__main__':
    main()