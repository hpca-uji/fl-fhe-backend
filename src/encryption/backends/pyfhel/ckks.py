from src.core.util.check_value import check_value
from src.encryption.base import EncryptionBase
from Pyfhel import Pyfhel, PyCtxt
import numpy as np


class PyFHELCKKS(EncryptionBase):
  def __init__(self):
    super(PyFHELCKKS, self).__init__()
    self.poly_modulus_degree=8192
    self.coef_mod_bit=[60, 40, 40, 60]
    self.global_scale=2**40
    self.context=None
    self.he = Pyfhel()

  def create_context(self):
    if not isinstance(self.coef_mod_bit, list) or not all(isinstance(x, int) for x in self.coef_mod_bit):
            raise ValueError(f"Invalid coefficient modulus bit sizes: {self.coef_mod_bit}. Must be a list of integers.")
    ckks_params = {
      'scheme': 'CKKS',
      'n': self.poly_modulus_degree,
      'scale': self.global_scale,
      'qi_sizes': self.coef_mod_bit
    }
    self.context = self.he.contextGen(**ckks_params)

  def get_context(self):
    return self.context

  def get_instance(self):
    return self.he

  def set_context(self, context):
    self.he.load_context(context)

  def set_poly_modulus_degree(self, degree=8192):
    check_value(degree, int, 'poly_modulus_degree', 'int')
    if degree not in [2048, 4096, 8192, 16384]:
      raise ValueError(f"Invalid poly_modulus_degree: {degree}. Must be a power of 2.")
    self.poly_modulus_degree = degree
    self.create_context()

  def set_coef_mod_bit(self, coefficient=None):
    if coefficient is not None and (not isinstance(coefficient, list) or not all(isinstance(x, int) for x in coefficient)):
      raise ValueError(f"Invalid coefficient modulus bit sizes: {coefficient}. Must be a list of integers.")
    self.coef_mod_bit = coefficient
    self.create_context()

  def set_global_scale(self, scale=2**40):
    check_value(scale, (int, float), 'global_scale', 'int or float')
    self.global_scale = scale
    self.create_context()

  def generate_keys(self, **kwargs):
    self.he.keyGen()

  def rotate_key_gen(self):
    self.he.rotateKeyGen()

  def set_private_key(self, private_key):
    check_value(private_key, (bytes), 'private_key', 'bytes')
    self.he.from_bytes_secret_key(private_key)

  def set_public_key(self, public_key):
    check_value(public_key, (bytes), 'private_key', 'bytes')
    self.he.from_bytes_public_key(public_key)

  def get_private_key(self):
    return self.context.serialize(save_secret_key = True)

  def get_public_key(self):
    return self.he.to_bytes_public_key()

  def encrypt(self, value):
    if isinstance(value, (int, float)):
      enc_text = self.he.encryptFrac(np.array([value]))
    elif isinstance(value, (np.ndarray, list, np.matrix)):
      if value.ndim == 1:
        enc_text = enc_text = np.array([self.he.encryptFrac(np.array([value[i]])) for i in range(value.shape[0])])
      elif value.ndim == 2:
        # matrix_size = value.shape
        enc_text = np.array([[self.he.encryptFrac(np.array([value[i, j]])) for j in range(value.shape[1])] for i in range(value.shape[0])])
    else:
      raise ValueError(f"Invalid value type: {type(value)}. Must be a numpy array, list, or numpy matrix.")

    return enc_text

  def decrypt(self, value):
    if isinstance(value, PyCtxt):
      return np.sum(self.he.decryptFrac(value))
    elif isinstance(value, (np.ndarray, list)):
      if value.ndim == 1:
        return np.array([np.sum(self.he.decryptFrac(value[i])) for i in range(value.shape[0])])
      else:
        return np.array([[np.sum(self.he.decryptFrac(value[i, j])) for j in range(value.shape[1])] for i in range(value.shape[0])])
        ## dec_text = np.array([self.he.decryptFrac(value[i]) for i in range(value.shape[0])])
    else:
      raise ValueError(f"Invalid value type: {type(value)}. Must be a PyCtxt.")

  def enc_dot(self, term1, term2):
    if isinstance(term1, (PyCtxt, np.ndarray)) and isinstance(term2, (PyCtxt, np.ndarray)):
      if term1.ndim == 1 and term2.ndim == 1:
        if term1.shape[0] != term2.shape[0]:
          raise ValueError(f"Invalid value size: 1: {term1.size()}, 2: {term2.size()} must have the same size.")
        return self.vec_dot(term1, term2)         
      elif term1.ndim == 1 and term2.ndim == 2:
        if term1.shape[0] != term2.shape[0]:
          raise ValueError(f"Invalid value size: 1: {term1.size()}, 2: {term2.size()} must have the same size.")
        return self.vec_matrix_dot(term1, term2)
      elif term1.shape == term2.shape:
        return self.matrix_dot(term1, term2)
      else:
        raise ValueError(f"Invalid value size: 1: {term1.size()}, 2: {term2.size()} must be a vector or a matrix.")
    else:
      raise ValueError(f"Invalid value type: 1: {type(term1)}, 2: {type(term2)}. Must be a PyCtxt.")

  def vec_dot(self, term1, term2):
    for i in range(term1.shape[0]):
      if i == 0:
        result = term1[i] * term2[i]
      else:
        result += term1[i] * term2[i]
    return result

  def vec_matrix_dot(self, term1, term2):
    enc_result = [None for _ in range(term2.shape[1])]
    for i in range(term2.shape[1]):
      for j in range(term1.shape[0]):
        if j==0:
          enc_result[i] = term1[j] * term2[j][i]
        else:
          enc_result[i] += term1[j] * term2[j][i]
    return np.array(enc_result)

  def matrix_dot(self, term1, term2):
    enc_result = [[None for _ in range(term1.shape[1])] for _ in range(term1.shape[0])]
    for i in range(term1.shape[0]):
      for j in range(term2.shape[1]):
        enc_result[i][j] = term1[i][0] * term2[0][j]
        for k in range(1, term1.shape[1]):
          temp_mul = term1[i][k] * term2[k][j]
          enc_result[i][j] += temp_mul
    return np.array(enc_result)

  '''
  def sum_encrypted_values(self, ctxt: PyCtxt, size: int) -> PyCtxt:
    """Homomorphically sums all slots in an encrypted ciphertext."""
    result = ctxt.copy()
    for i in range(1, size):
        rotated_ctxt = ctxt.copy()
        self.he.rotate(rotated_ctxt, i)
        result += rotated_ctxt
    return result
  '''

  def enc_add(self, term1, term2, **kwargs):
    if isinstance(term1, PyCtxt) and isinstance(term1, PyCtxt):
      return term1 + term2
    if isinstance(term1, np.array) and isinstance(term1, np.array):
      if term1.ndim == 1 and term2 == 1:
        return [term1[i] + term2[i] for i in range(term1.shape[0])]
      elif term1.ndim == 2 and term2 == 1:
        return [[term1[i, j]+ term2[j] for j in range(term2.shape[0])] for i in range(term1.shape[0])]
      elif term1.ndim == 2 and term2.ndim == 2:
        if term1.shape == term2.shape:
          return [[term1[i][j]+ term2[i][j] for j in range(term1.shape[1])] for i in range(term1.shape[0])]
        else:
          raise ValueError(f"Both matrices must have the same dimensions.")
      else:
        raise ValueError(f"Error: Please check the vectors or matrices dimensions.")
    else:
      raise ValueError(f"Invalid value type: 1: {type(term1)}, 2: {type(term2)}. Must be a PyCtxt, np.array or np.matrix.")

  def enc_matmul(self, term1, term2):
    return self.enc_dot(term1, term2)
