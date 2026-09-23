import time
from src.core.util.check_value import check_value
from src.encryption.base import EncryptionBase
from Pyfhel import Pyfhel, PyCtxt
import numpy as np


class PyFHELCKKS(EncryptionBase):
  def __init__(self):
    super(PyFHELCKKS, self).__init__()
    self.poly_modulus_degree=8192
    self.coef_mod_bit=[60, 40, 60]
    self.global_scale=2**30
    self.context=None
    self.he = Pyfhel()

  def create_context(self):
    if not isinstance(self.coef_mod_bit, list) or not all(isinstance(x, int) for x in self.coef_mod_bit):
            raise ValueError(f"Invalid coefficient modulus bit sizes: {self.coef_mod_bit}. Must be a list of integers.")
    ckks_params = {
      'scheme': 'CKKS',
      'n': self.poly_modulus_degree,
      'scale': self.global_scale,
      'qi_sizes': self.coef_mod_bit,
      'sec': 128 if self.poly_modulus_degree <= 32768 else 0
    }
    print("Creating PyFHEL CKKS context with parameters:", ckks_params)
    self.context = self.he.contextGen(**ckks_params)

  def get_context(self):
    return {
      "public_key": self.he.to_bytes_public_key(),
      "relin_key": self.he.to_bytes_relin_key(),
      "rotate_key": self.he.to_bytes_rotate_key(),
      "context": self.he.to_bytes_context()
    }

  def get_instance(self):
    return self.he

  def set_context(self, context):
    if isinstance(context, dict):
      if 'context' in context:
        self.he.from_bytes_context(context['context'])
      if 'public_key' in context:
        self.he.from_bytes_public_key(context['public_key'])
      if 'relin_key' in context:
        self.he.from_bytes_relin_key(context['relin_key'])
      if 'rotate_key' in context:
        self.he.from_bytes_rotate_key(context['rotate_key'])
    else:
      self.he.from_bytes_context(context)
    return self.he

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
    self.he.keyGen()
    self.he.relinKeyGen()
    self.he.rotateKeyGen()
    self.keygen_time = time.perf_counter() - t0
    return self.keygen_time

  def rotate_key_gen(self):
    self.he.rotateKeyGen()

  def set_private_key(self, private_key):
    if isinstance(private_key, dict):
      if 'context' in private_key:
        self.he.from_bytes_context(private_key['context'])
      if 'public_key' in private_key:
        self.he.from_bytes_public_key(private_key['public_key'])
      if 'secret_key' in private_key:
        self.he.from_bytes_secret_key(private_key['secret_key'])
      if 'relin_key' in private_key:
        self.he.from_bytes_relin_key(private_key['relin_key'])
      if 'rotate_key' in private_key:
        self.he.from_bytes_rotate_key(private_key['rotate_key'])
    else:
      check_value(private_key, (bytes), 'private_key', 'bytes')
      self.he.from_bytes_secret_key(private_key)

  def set_public_key(self, public_key):
    check_value(public_key, (bytes), 'private_key', 'bytes')
    self.he.from_bytes_public_key(public_key)

  def get_private_key(self):
    return {
      "secret_key": self.he.to_bytes_secret_key(), 
      "public_key": self.he.to_bytes_public_key(), 
      "relin_key": self.he.to_bytes_relin_key(),
      "rotate_key": self.he.to_bytes_rotate_key(),
      "context": self.he.to_bytes_context()
    }

  def get_public_key(self):
    return self.he.to_bytes_public_key()

  def encrypt(self, value, **kwargs):
    if isinstance(value, (int, float, np.number)):
      return self.he.encryptFrac(np.array([value], dtype=np.float64))
    elif isinstance(value, (np.ndarray, list, np.matrix)):
      # Flatten array of any dimension
      flat_value = np.array(value, dtype=np.float64).flatten()
      max_slots = self.poly_modulus_degree // 2
      
      try:
          if len(flat_value) <= max_slots:
              return self.he.encryptFrac(flat_value)
      except Exception:
          # Fallback: if Pyfhel raises an error (like ArithmeticError for nSlots),
          # it implies the actual context slots are smaller than expected.
          # We'll use 2048 as a safe minimum fallback for chunking.
          max_slots = 2048

      # Chunk the array if it exceeds the maximum slots or if the fallback triggered
      chunks = [flat_value[i:i + max_slots] for i in range(0, len(flat_value), max_slots)]
      return np.array([self.he.encryptFrac(chunk) for chunk in chunks], dtype=object)
    else:
      raise ValueError(f"Invalid value type: {type(value)}. Must be a numpy array, list, or numpy matrix.")

  def decrypt(self, value, **kwargs):
    if isinstance(value, PyCtxt):
      return self.he.decryptFrac(value)
    elif isinstance(value, (np.ndarray, list)):
      value = np.array(value, dtype=object)
      if value.size > 0 and isinstance(value.flatten()[0], PyCtxt):
          decrypted_chunks = [self.he.decryptFrac(c) for c in value.flatten()]
          return np.concatenate(decrypted_chunks)
      return value
    elif isinstance(value, (float, int, np.number)):
      return float(value)
    else:
      raise ValueError(f"Invalid value type for decryption: {type(value)}. Expected PyCtxt.")

  def enc_dot(self, term1, term2, **kwargs):
    is_term1_vec = isinstance(term1, PyCtxt)
    is_term2_vec = isinstance(term2, PyCtxt)

    if is_term1_vec and is_term2_vec:
        return self.vec_dot(term1, term2)
    if not is_term1_vec and is_term2_vec:
        return np.array([self.vec_dot(row, term2) for row in term1], dtype=object)
    if is_term1_vec and not is_term2_vec:
        return np.array([self.vec_dot(term1, col) for col in term2], dtype=object)
    if not is_term1_vec and not is_term2_vec:
        num_rows1 = len(term1)
        num_cols2 = len(term2)
        return np.array([[self.vec_dot(term1[i], term2[j]) for j in range(num_cols2)] for i in range(num_rows1)], dtype=object)

  def vec_dot(self, term1, term2):
    result = term1 * term2
    if hasattr(self.he, 'cumul_add'):
        return self.he.cumul_add(result, in_new_ctxt=True)
    else:
        rot_result = result.copy()
        for i in range(int(np.log2(self.poly_modulus_degree // 2))):
            temp = rot_result.copy()
            self.he.rotate(temp, 2**i)
            rot_result += temp
        return rot_result

  def enc_add(self, term1, term2, **kwargs):
    is_term1_vec = isinstance(term1, PyCtxt)
    is_term2_vec = isinstance(term2, PyCtxt)

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
    # Element-wise multiplication, with automatic rescaling for PyFHEL
    res = term1 * term2
    if isinstance(res, np.ndarray):
        def _rescale(c):
            if isinstance(c, PyCtxt):
                # ~ performs rescale_to_next() in PyFHEL, dropping the scale safely
                ~c
            return c
        # Apply rescale to all ciphertexts in the array chunks
        return np.vectorize(_rescale, otypes=[object])(res)
    else:
        if isinstance(res, PyCtxt):
            ~res
        return res

  def serialize(self, value):
    if isinstance(value, PyCtxt):
      return value.to_bytes()
    elif isinstance(value, (np.ndarray, list)):
      value = np.array(value, dtype=object)
      flat_val = value.flatten()
      ser_val = [c.to_bytes() if hasattr(c, "to_bytes") else c for c in flat_val]
      return np.array(ser_val, dtype=object).reshape(value.shape)
    return value

  def deserialize(self, value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
      value = value.item()
    if isinstance(value, bytes):
      ctxt = PyCtxt(pyfhel=self.he)
      ctxt.from_bytes(bytes(value))
      return ctxt
    elif isinstance(value, (np.ndarray, list)):
      value = np.array(value, dtype=object)
      flat_val = value.flatten()
      deser_val = []
      for b in flat_val:
        if isinstance(b, (bytes, np.bytes_)):
          c = PyCtxt(pyfhel=self.he)
          c.from_bytes(bytes(b))
          deser_val.append(c)
        else:
          deser_val.append(b)
      return np.array(deser_val, dtype=object).reshape(value.shape)
    return value
