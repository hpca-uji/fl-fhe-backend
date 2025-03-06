from openfhe import *
import numpy as np
from .common.enum.secret_key_dist import SecretKeyDist
from .common.enum.scaling_technique import ScalingTechnique
from .common.enum.security_level import SecurityLevel
from .common.enum.pke_schema_feature import PKESchemaFeature
from src.encryption.base import EncryptionBase

class OpenFHECKKS(EncryptionBase):
  def __init__(self):
    super(OpenFHECKKS, self).__init__()
    self.poly_modulus_degree=8192
    self.coef_mod_bit=[60, 40, 40, 60]
    self.global_scale=40
    self.secret_key_dist=SecretKeyDist.GAUSSIAN
    self.security_level=SecurityLevel.HEStd_128_classic
    self.ring_dim=(1 << 12)
    self.first_mode_size=1
    self.context=None
    self.scaling_technique=ScalingTechnique.FIXED
    self.parameters=CCParamsCKKSRNS()
    self.keys=None
  
    
  def set_secret_key_dist(self, secret_key_dist):
    self.secret_key_dist=secret_key_dist
    self.parameters.SetSecretKeyDist(secret_key_dist)
    
  def set_ring_dim(self, ring_dim):
    self.ring_dim=ring_dim
    self.parameters.SetRingDim(ring_dim)
    
  def set_first_mode_size(self, first_mode_size):
    self.first_mode_size=first_mode_size
    self.parameters.SetFirstModSize(first_mode_size)
    
  def set_scaling_technique(self, scaling_technique):
    self.scaling_technique=scaling_technique
    self.parameters.SetScalingTechnique(scaling_technique)
    
  def set_global_scale(self, global_scale):
    self.global_scale=global_scale
    self.parameters.SetScalingModSize(global_scale)
    
  def set_security_level(self, security_level):
    self.security_level=security_level
    self.parameters.SetSecurityLevel(security_level)
    
  def create_context(self):
    self.context = CKKSEncryptionContext(self.parameters)
    
  def set_depth(self, depth):
    self.parameters.SetMultiplicativeDepth(depth)
    
  def set_context_feature(self, context_feature: PKESchemaFeature):
    self.context.Enable(context_feature)
  
  def eval_bootstrap_setup(self, level_budget, bsgs_dim, num_slots):
    self.context.EvalBootstrapSetup(level_budget, bsgs_dim, num_slots)
    
  def make_ckks_packed_plain_text(self, value, size: int = None, level: int = None, params=None, slots: int = None):
    if isinstance(value, (int, float)):
      enc_text = self.context.MakeCKKSPackedPlaintext(np.array([value]), size, level, params, slots)
    elif isinstance(value, (np.ndarray, list, np.matrix)):
      if value.ndim == 1:
        enc_text = enc_text = np.array([self.context.MakeCKKSPackedPlaintext(np.array([value[i]]), size, level, params, slots) for i in range(value.shape[0])])
      elif value.ndim == 2:
        matrix_size = value.shape
        enc_text = np.array([[self.context.MakeCKKSPackedPlaintext(np.array([value[i, j]]), size, level, params, slots) for j in range(value.shape[1])] for i in range(value.shape[0])])
    else:
      raise ValueError(f"Invalid value type: {type(value)}. Must be a numpy array, list, or numpy matrix.")

    return enc_text
    
  def eval_mult_key_gen(self):
    self.context.EvalMultKeyGen()
  
  def get_context(self):
    return self.context
    
  def generate_keys(self, **kwargs):
    self.keys=self.context.KeyGen()
    
  def get_private_key(self):
    return self.keys.secretKey
    
  def get_public_key(self):
    pass
    
  def encrypt(self, value, **kwargs):
    if isinstance(value, (np.ndarray, list, np.matrix)):
      if value.ndim == 1:
        enc_text = enc_text = np.array([self.context.Encrypt(np.array([value[i]])) for i in range(value.shape[0])])
      elif value.ndim == 2:
        matrix_size = value.shape
        enc_text = np.array([[self.he.Encrypt(np.array([value[i, j]])) for j in range(value.shape[1])] for i in range(value.shape[0])])
    else:
      raise ValueError(f"Invalid value type: {type(value)}. Must be a numpy array, list, or numpy matrix.")

    return enc_text
    
  def decrypt(self, value, **kwargs):
    if isinstance(value, (np.ndarray, list, np.matrix)):
      if value.ndim == 1:
        enc_text = enc_text = np.array([self.context.Encrypt(np.array([value[i]])) for i in range(value.shape[0])])
      elif value.ndim == 2:
        matrix_size = value.shape
        enc_text = np.array([[self.he.Encrypt(np.array([value[i, j]])) for j in range(value.shape[1])] for i in range(value.shape[0])])
    else:
      raise ValueError(f"Invalid value type: {type(value)}. Must be a numpy array, list, or numpy matrix.")

    return enc_text
    
    
  def enc_dot(self, term1, term2, **kwargs):
    if isinstance(term1, (np.ndarray)) and isinstance(term2, (np.ndarray)):
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
      raise ValueError(f"Invalid value type: 1: {type(term1)}, 2: {type(term2)} must be a numpy array or matrix.")
    
  def vec_dot(self, term1, term2):
    for i in range(term1.shape[0]):
      if i == 0:
        result = self.context.EvalMult(term1[i], term2[i])
      else:
        result = self.context.EvalAdd(result, self.context.EvalMult(term1[i], term2[i]))
    return result

  def vec_matrix_dot(self, term1, term2):
    enc_result = [None for _ in range(term2.shape[1])]
    for i in range(term2.shape[1]):
        for j in range(term1.shape[0]):
          if j==0:
            enc_result[i] = self.context.EvalMult(term1[j], term2[j][i])
          else:
            enc_result[i] = self.context.EvalAdd(enc_result[i], self.context.EvalMult(term1[j], term2[j][i]))
    return enc_result

  def matrix_dot(self, term1, term2):
    enc_result = [[None for _ in range(term1.shape[1])] for _ in range(term1.shape[0])]

    for i in range(term1.shape[0]):
        for j in range(term2.shape[1]):
            enc_result[i][j] = self.context.EvalMult(term1[i][0], term2[0][j])
            for k in range(1, term1.shape[1]):
                temp_mul = self.context.EvalMult(term1[i][k], term2[k][j])
                enc_result[i][j] = self.context.EvalAdd(enc_result[i][j], temp_mul)
    return enc_result
    
  def enc_add(self, term1, term2, **kwargs):
    # TODO(DQ) Verify the types
    if isinstance(term1, np.array) and isinstance(term1, np.array):
      return self.context.EvalAdd(term1, term2)
    if isinstance(term1, np.array) and isinstance(term1, np.array):
      if term1.ndim == 1 and term2 == 1:
        return [ self.context.EvalAdd(term1[i], term2[i]) for i in range(term1.shape[0])]
      elif term1.ndim == 2 and term2 == 1:
        return [[ self.context.EvalAdd(term1[i, j], term2[j]) for j in range(term2.shape[0])] for i in range(term1.shape[0])]
      elif term1.ndim == 2 and term2.ndim == 2:
        if term1.shape == term2.shape:
          return [[self.context.EvalAdd(term1[i][j], term2[i][j]) for j in range(term1.shape[1])] for i in range(term1.shape[0])]
        else:
          raise ValueError(f"Both matrices must have the same dimensions.")
      else:
        raise ValueError(f"Error: Please check the vectors or matrices dimensions.")
    else:
      raise ValueError(f"Invalid value type: 1: {type(term1)}, 2: {type(term2)}. Must be a PyCtxt, np.array or np.matrix.")
    
    
  def enc_matmul(self, term1, term2, **kwargs):
    return self.enc_dot(term1, term2)
    
  def vec_dot(self, term1, term2):
    for i in range(term1.shape[0]):
      if i == 0:
        result = self.context.EvalMult(term1[i], term2[i])
      else:
        result = self.context.EvalAdd(result, self.context.EvalMult(term1[i], term2[i]))
    return result

  def vec_matrix_dot(self, term1, term2):
    enc_result = [None for _ in range(term2.shape[1])]
    for i in range(term2.shape[1]):
        for j in range(term1.shape[0]):
          if j==0:
            enc_result[i] = self.context.EvalMult(term1[j], term2[j][i])
          else:
            enc_result[i] = self.context.EvalAdd(enc_result[i], self.context.EvalMult(term1[j], term2[j][i]))
    return enc_result

  def matrix_dot(self, term1, term2):
    enc_result = [[None for _ in range(term1.shape[1])] for _ in range(term1.shape[0])]

    for i in range(term1.shape[0]):
        for j in range(term2.shape[1]):
            enc_result[i][j] = term1[i][0] * term2[0][j]
            for k in range(1, term1.shape[1]):
                temp_mul = term1[i][k] * term2[k][j]
                enc_result[i][j] += temp_mul
    return enc_result