import time
from src.core.util.check_value import check_value
from src.encryption.base import EncryptionBase
import tenseal as ts
import numpy as np

class TensealCKKS(EncryptionBase):
    def __init__(self):
        super(TensealCKKS, self).__init__()
        self.poly_modulus_degree=4096
        self.coef_mod_bit=[40, 20, 40]
        self.global_scale=2**30 
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
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        return self.context
    
    def get_context(self):
        return self.context.serialize()
    
    def set_context(self, context):
        if isinstance(context, dict) and 'context' in context:
            self.context = ts.context_from(context['context'])
        else:
            self.context = ts.context_from(context)
        return self.context
    
    def set_poly_modulus_degree(self, degree=8192):
        check_value(degree, int, 'poly_modulus_degree', 'int')
        if degree not in [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
            raise ValueError(f"Invalid poly_modulus_degree: {degree}. Must be a power of 2.")
    
        self.poly_modulus_degree = degree
    
    def set_coef_mod_bit(self, coefficient=None):
        if coefficient is not None and (not isinstance(coefficient, list) or not all(isinstance(x, int) for x in coefficient)):
            raise ValueError(f"Invalid coefficient modulus bit sizes: {coefficient}. Must be a list of integers.")
        self.coef_mod_bit = coefficient
        
    def set_global_scale(self, scale=2**40):
        check_value(scale, (int, float), 'global_scale', 'int or float')
        self.global_scale = scale

    def generate_keys(self, **kwargs):
        t0 = time.perf_counter()
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        self.keygen_time = time.perf_counter() - t0
        return self.keygen_time
    
    def set_private_key(self, private_key):
        if isinstance(private_key, dict) and 'context' in private_key:
            self.context = ts.context_from(private_key['context'])
        else:
            self.context = ts.context_from(private_key)
    
    def set_public_key(self, public_key):
        if isinstance(public_key, dict) and 'context' in public_key:
            self.context = ts.context_from(public_key['context'])
        else:
            self.context = ts.context_from(public_key)
    
    def get_private_key(self):
        return self.context.serialize(save_secret_key = True)
    
    def get_public_key(self):
        self.context.make_context_public()
        return self.context.serialize()
    
    def plain_tensor(self, value):
        return ts.plain_tensor(value)
    
    def serialize(self, value):
        if isinstance(value, ts.CKKSVector):
            return value.serialize()
        elif isinstance(value, (np.ndarray, list)):
            value = np.array(value, dtype=object)
            flat_val = value.flatten()
            ser_val = [c.serialize() if hasattr(c, "serialize") else c for c in flat_val]
            return np.array(ser_val, dtype=object).reshape(value.shape)
        return value
    
    def deserialize(self, value):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        if isinstance(value, bytes):
            return ts.ckks_vector_from(self.context, bytes(value))
        elif isinstance(value, (np.ndarray, list)):
            value = np.array(value, dtype=object)
            flat_val = value.flatten()
            deser_val = []
            for b in flat_val:
                if isinstance(b, (bytes, np.bytes_)):
                    deser_val.append(ts.ckks_vector_from(self.context, bytes(b)))
                else:
                    deser_val.append(b)
            return np.array(deser_val, dtype=object).reshape(value.shape)
        return value
    
    def encrypt(self, value, **kwargs):
        if isinstance(value, (int, float, np.number)):
            return ts.ckks_vector(self.context, [float(value)])

        # Flatten array of any dimension
        flat_value = np.array(value, dtype=np.float64).flatten().tolist()
        max_slots = self.poly_modulus_degree // 2
        
        if len(flat_value) <= max_slots:
            return ts.ckks_vector(self.context, flat_value)
        else:
            chunks = [flat_value[i:i + max_slots] for i in range(0, len(flat_value), max_slots)]
            return np.array([ts.ckks_vector(self.context, chunk) for chunk in chunks], dtype=object)
    
    def decrypt(self, value, **kwargs):
        if isinstance(value, ts.CKKSVector):
            dec = value.decrypt()
            return dec[0] if len(dec) == 1 else np.array(dec, dtype=np.float64)
        elif isinstance(value, (np.ndarray, list)):
            value = np.array(value, dtype=object)
            if value.size > 0 and isinstance(value.flatten()[0], ts.CKKSVector):
                decrypted_chunks = [c.decrypt() for c in value.flatten()]
                return np.concatenate(decrypted_chunks)
            return value
        else:
            raise ValueError(f"Invalid value type for decryption: {type(value)}. Expected CKKSVector.")
            
    def plain_vector(self, value):
        shape = value.shape
        return ts.enc_matmul_encoding(self.context, value)
        
    def enc_dot(self, term1, term2, **kwargs):
        is_term1_vec = isinstance(term1, ts.CKKSVector)
        is_term2_vec = isinstance(term2, ts.CKKSVector)

        if is_term1_vec and is_term2_vec:
            return term1.dot(term2)
        if not is_term1_vec and is_term2_vec:
            return np.array([row.dot(term2) for row in term1], dtype=object)
        if is_term1_vec and not is_term2_vec:
            return np.array([term1.dot(col) for col in term2], dtype=object)
        if not is_term1_vec and not is_term2_vec:
            num_rows1 = len(term1)
            num_cols2 = len(term2)
            return np.array([[term1[i].dot(term2[j]) for j in range(num_cols2)] for i in range(num_rows1)], dtype=object)
        
    def enc_add(self, term1, term2, **kwargs):
        is_term1_vec = isinstance(term1, ts.CKKSVector)
        is_term2_vec = isinstance(term2, ts.CKKSVector)

        if is_term1_vec and is_term2_vec:
            return term1 + term2
        if not is_term1_vec and not is_term2_vec:
            if term1.shape != term2.shape:
                raise ValueError("Matrices must have the same dimensions for addition.")
            return np.array([term1[i] + term2[i] for i in range(len(term1))], dtype=object)
        if not is_term1_vec and is_term2_vec:
            return np.array([row + term2 for row in term1], dtype=object)
        if is_term1_vec and not is_term2_vec:
            return np.array([term1 + row for row in term2], dtype=object)
    
    def enc_matmul(self, term1, term2, **kwargs):
        return self.enc_dot(term1, term2)
        
    def enc_mul(self, term1, term2, **kwargs):
        if isinstance(term1, np.ndarray) and isinstance(term2, (int, float, np.number)):
            # Multiply each chunk by the scalar
            flat_res = [c * float(term2) if isinstance(c, ts.CKKSVector) else c * term2 for c in term1.flatten()]
            return np.array(flat_res, dtype=object).reshape(term1.shape)
        if isinstance(term1, ts.CKKSVector) and isinstance(term2, (int, float, np.number)):
            return term1 * float(term2)
        return term1 * term2