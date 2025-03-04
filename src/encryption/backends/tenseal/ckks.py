from src.core.util.check_value import check_value
from src.encryption.base import EncryptionBase
import tenseal as ts
import numpy as np

class TensealCKKS(EncryptionBase):
    def __init__(self):
        super(TensealCKKS, self).__init__()
        self.poly_modulus_degree=8192
        self.coef_mod_bit=[60, 40, 40, 60]
        self.global_scale=2**40 
        self.context=None

    def create_context(self):
        if not isinstance(self.coef_mod_bit, list) or not all(isinstance(x, int) for x in self.coef_mod_bit):
            raise ValueError(f"Invalid coefficient modulus bit sizes: {self.coef_mod_bit}. Must be a list of integers.")

        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS, 
            poly_modulus_degree=self.poly_modulus_degree, 
            coeff_mod_bit_sizes=self.coef_mod_bit
        )
        self.context.global_scale = self.global_scale
        return self.context
    
    def get_context(self):
        return self.context.serialize()
    
    def set_context(self, context):
        return ts.context_from(context)
    
    def set_poly_modulus_degree(self, degree=8192):
        check_value(degree, int, 'poly_modulus_degree', 'int')
        if degree not in [2048, 4096, 8192, 16384]:
            raise ValueError(f"Invalid poly_modulus_degree: {degree}. Must be a power of 2.")
    
        self.poly_modulus_degree = degree
        return self.create_context()
    
    def set_coef_mod_bit(self, coefficient=None):
        if coefficient is not None and (not isinstance(coefficient, list) or not all(isinstance(x, int) for x in coefficient)):
            raise ValueError(f"Invalid coefficient modulus bit sizes: {coefficient}. Must be a list of integers.")
        self.coef_mod_bit = coefficient
        return self.create_context()
        
    def set_global_scale(self, scale=2**40):
        check_value(scale, (int, float), 'global_scale', 'int or float')
        self.global_scale = scale
        return self.create_context()

    def generate_keys(self, **kwargs):
        self.context.generate_galois_keys()
    
    def set_private_key(self, private_key):
        ts.context_from(private_key)
    
    def set_public_key(self, public_key):
        ts.context_from(public_key)
    
    def get_private_key(self):
        return self.context.serialize(save_secret_key = True)
    
    def get_public_key(self):
        self.context.make_context_public()
        return self.context.serialize()
    
    def plain_tensor(self, value):
        return ts.plain_tensor(value)
    
    def serialize(self, value):
        return 
    
    def deserialize(self, value):
        return 
    
    def encrypt(self, value):
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                cipher_serial = ts.ckks_vector(self.context, value)
            elif value.ndim == 2:
                matrix_size = value.shape
                matrix_plain = ts.plain_tensor(value.flatten(), matrix_size)
                cipher_serial = ts.ckks_tensor(self.context, matrix_plain)
        elif isinstance(value, list):
            cipher_serial = ts.ckks_vector(self.context, value)
            
        elif isinstance(value, np.matrix):
            matrix_size = value.shape
            matrix_plain = ts.plain_tensor(value.flatten(), matrix_size)
            cipher_serial = ts.ckks_tensor(self.context, matrix_plain) 
        else:
            raise ValueError(f"Invalid value type: {type(value)}. Must be a numpy array, list, or numpy matrix.")
        
        return cipher_serial
    
    def decrypt(self, value):
        if isinstance(value, ts.CKKSVector):
            return value.decrypt()
        elif isinstance(value, ts.CKKSTensor):
            return value.decrypt().tolist()
        else:
            raise ValueError(f"Invalid value type: {type(value)}. Must be a CKKSVector or PlainTensor.")
            
    def plain_vector(self, value):
        shape = value.shape
        return ts.enc_matmul_encoding(self.context, value)
        
    def enc_dot(self, term1, term2):
        if not (isinstance(term1, ts.CKKSVector) | isinstance(term1, ts.CKKSTensor) | isinstance(term2, ts.CKKSVector) 
                | isinstance(term2, ts.CKKSTensor) | isinstance(term2, ts.CKKSVector)):
            raise ValueError(f"Invalid value type: 1: {type(term1)}, 2: {type(term2)}. Must be a CKKSVector or PlainTensor.")
        
        return term1.dot(term2)
        
        
    def enc_add(self, term1, term2):
        if not (isinstance(term1, ts.CKKSVector) | isinstance(term1, ts.CKKSTensor) | isinstance(term2, ts.CKKSVector) 
                | isinstance(term2, ts.CKKSTensor) | isinstance(term2, ts.CKKSVector)):
            raise ValueError(f"Invalid value type: 1: {type(term1)}, 2: {type(term2)}. Must be a CKKSVector or PlainTensor.")
        
        return term1.add(term2)
    
    def enc_matmul(self, term1, term2):
        if isinstance(term1, ts.CKKSVector) and isinstance(term2, ts.PlainTensor):
            result = term1.matmul(term2)
        elif isinstance(term1, ts.CKKSTensor) and isinstance(term2, ts.PlainTensor):
            result = term1.mm(term2)
        else:
            raise ValueError(f"Invalid value type: 1: {type(term1)}, 2: {type(term2)}. Must be a CKKSVector or PlainTensor.")
        
        return result