from src.encryption.factory import HomomorphicEncrytionFactory
import numpy as np

def main():
    he_backend = HomomorphicEncrytionFactory.get_backend(library='PYFHEL', schema='CKKS')
    he_backend.set_global_scale(2**30)
    he_backend.set_poly_modulus_degree(2**14)
    he_backend.set_coef_mod_bit([60, 30, 30, 30, 60] )
    he_backend.generate_keys()
    he_backend.create_context()
    
    # Vector encryption and decryption
    vector = np.array([1.5, 2.6, 3.8])
    encrypted_vector = he_backend.encrypt(vector)
    dencrypted_vector = he_backend.decrypt(encrypted_vector)
    print(dencrypted_vector)


    # Matrix
    matrix = np.matrix([[1.1, 2.2], [3.3, 4.4]])
    encrypted_matrix = he_backend.encrypt(matrix)
    dencrypted_matrix = he_backend.decrypt(encrypted_matrix)
    print(dencrypted_matrix)
    
    
    vector1 = np.array([1.5, 2.6, 3.8])
    vector2 = np.array([1.5, 2.6, 3.8])
    
    # Vector dot product
    encrypted_vector1 = he_backend.encrypt(vector1)
    encrypted_vector2 = he_backend.encrypt(vector2)
    dot_product = he_backend.enc_dot(encrypted_vector1, encrypted_vector2)
    dencrypted_vector = he_backend.decrypt(dot_product)
    
    matrix1 = np.matrix([[1.1, 2.2], [3.3, 4.4]])
    matrix2 = np.matrix([[1.1, 2.2], [3.3, 4.4]])
    
    # Matrix dot product
    encrypted_matrix1 = he_backend.encrypt(matrix1)
    encrypted_matrix2 = he_backend.encrypt(matrix2)
    dot_matrix = he_backend.enc_dot(encrypted_matrix1, encrypted_matrix2)
    dencrypted_dot_matrix = he_backend.decrypt(dot_matrix)
    print(dencrypted_dot_matrix)
    
    
if __name__ == '__main__':
    main()