import time
from openfhe import *
import numpy as np
import os
import tempfile
from .common.enum.secret_key_dist import SecretKeyDist
from .common.enum.scaling_technique import ScalingTechnique
from .common.enum.security_level import SecurityLevel
from .common.enum.pke_schema_feature import PKESchemaFeature
from src.encryption.base import EncryptionBase
from src.core.util.check_value import check_value

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
    self.scaling_technique=ScalingTechnique.FLEXIBLEAUTO
    self.parameters=CCParamsCKKSRNS()
    self.keys=None
  
  def set_poly_modulus_degree(self, degree=8192):
    check_value(degree, int, 'poly_modulus_degree', 'int')
    if degree not in [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
        raise ValueError(f"Invalid poly_modulus_degree: {degree}. Must be a power of 2.")
    self.poly_modulus_degree = degree
    # OpenFHE cyclotomic order (ring_dim) is typically 2 * poly_modulus_degree
    self.set_ring_dim(degree * 2)
    
  def set_coef_mod_bit(self, coefficient=None):
    if coefficient is not None and (not isinstance(coefficient, list) or not all(isinstance(x, int) for x in coefficient)):
        raise ValueError(f"Invalid coefficient modulus bit sizes: {coefficient}. Must be a list of integers.")
    self.coef_mod_bit = coefficient
    if coefficient and len(coefficient) > 0:
        self.set_first_mode_size(coefficient[0])
        if len(coefficient) > 1:
            self.set_global_scale(coefficient[1])
            
  def _serialize_to_bytes(self, obj, obj_type):
      fd, temp_path = tempfile.mkstemp()
      os.close(fd)
      try:
          if obj_type in ["Context", "PublicKey", "SecretKey", "Ciphertext"]:
              res = SerializeToFile(temp_path, obj, BINARY)
          elif obj_type == "MultKey":
              res = self.context.SerializeEvalMultKey(temp_path, BINARY)
          elif obj_type == "RotKey":
              res = self.context.SerializeEvalAutomorphismKey(temp_path, BINARY)
          else:
              raise ValueError(f"Unknown obj_type: {obj_type}")
              
          if not res:
              raise RuntimeError("OpenFHE serialization failed.")
          with open(temp_path, "rb") as f:
              return f.read()
      finally:
          if os.path.exists(temp_path):
              os.remove(temp_path)
              
  def _deserialize_from_bytes(self, data, obj_type):
      fd, temp_path = tempfile.mkstemp()
      os.close(fd)
      try:
          with open(temp_path, "wb") as f:
              f.write(data)
              
          if obj_type == "Context":
              obj, res = DeserializeCryptoContext(temp_path, BINARY)
          elif obj_type == "PublicKey":
              obj, res = DeserializePublicKey(temp_path, BINARY)
          elif obj_type == "SecretKey":
              obj, res = DeserializePrivateKey(temp_path, BINARY)
          elif obj_type == "Ciphertext":
              obj, res = DeserializeCiphertext(temp_path, BINARY)
          elif obj_type == "MultKey":
              res = self.context.DeserializeEvalMultKey(temp_path, BINARY)
              obj = True
          elif obj_type == "RotKey":
              res = self.context.DeserializeEvalAutomorphismKey(temp_path, BINARY)
              obj = True
          else:
              raise ValueError(f"Unknown obj_type: {obj_type}")
              
          if not res:
              raise RuntimeError(f"OpenFHE deserialization failed for {obj_type}.")
          return obj
      finally:
          if os.path.exists(temp_path):
              os.remove(temp_path)

  def set_context(self, context):
    if isinstance(context, dict) and 'context' in context:
        # Release global contexts in OpenFHE to avoid KeyTag collision during deserialization
        ReleaseAllContexts()
        self.context = self._deserialize_from_bytes(context['context'], "Context")
        self.context.Enable(PKE)
        self.context.Enable(KEYSWITCH)
        self.context.Enable(LEVELEDSHE)
        self.context.Enable(ADVANCEDSHE)
        if 'public_key' in context:
            pk = self._deserialize_from_bytes(context['public_key'], "PublicKey")
            if self.keys is None:
                class _DummyKeys: pass
                self.keys = _DummyKeys()
                self.keys.secretKey = None
            self.keys.publicKey = pk
        if 'mult_key' in context:
            # Clear global multiplication keys for the matching tags
            ClearEvalMultKeys()
            self._deserialize_from_bytes(context['mult_key'], "MultKey")
        if 'rot_key' in context:
            # Clear context-specific rotation keys
            self.context.ClearEvalAutomorphismKeys()
            self._deserialize_from_bytes(context['rot_key'], "RotKey")
    else:
        self.context = context
        
  def set_private_key(self, private_key):
    if isinstance(private_key, dict):
        self.set_context(private_key)
        if 'secret_key' in private_key:
            sk = self._deserialize_from_bytes(private_key['secret_key'], "SecretKey")
            self.keys.secretKey = sk
    
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
    self.context = GenCryptoContext(self.parameters)
    self.context.Enable(PKE)
    self.context.Enable(KEYSWITCH)
    self.context.Enable(LEVELEDSHE)
    self.context.Enable(ADVANCEDSHE)
    
  def set_depth(self, depth):
    self.parameters.SetMultiplicativeDepth(depth)
    
  def set_context_feature(self, context_feature: PKESchemaFeature):
    self.context.Enable(context_feature)
  
  def eval_bootstrap_setup(self, level_budget, bsgs_dim, num_slots):
    self.context.EvalBootstrapSetup(level_budget, bsgs_dim, num_slots)
    
  def make_ckks_packed_plain_text(self, value, size: int = None, level: int = None, params=None, slots: int = None):
    if isinstance(value, (int, float)):
      # Pack a single value as a vector of length 1
      packed_value = self.context.MakeCKKSPackedPlaintext([value], size, level, params, slots)
    elif isinstance(value, (np.ndarray, list)):
      value = np.array(value) # ensure it's a numpy array
      if value.ndim == 1:
        # Pack a vector
        packed_value = self.context.MakeCKKSPackedPlaintext(value.tolist(), size, level, params, slots)
      elif value.ndim == 2:
        # Pack each row of the matrix as a separate plaintext
        packed_value = np.array([self.context.MakeCKKSPackedPlaintext(row.tolist(), size, level, params, slots) for row in value])
      else:
        raise ValueError(f"Unsupported numpy array ndim: {value.ndim}. Only 1D and 2D arrays are supported.")
    else:
      raise ValueError(f"Invalid value type: {type(value)}. Must be an int, float, numpy array, or list.")

    return packed_value
    
  def eval_mult_key_gen(self):
    self.context.EvalMultKeyGen(self.keys.secretKey)
  
  def get_context(self):
    return {
        "context": self._serialize_to_bytes(self.context, "Context"),
        "public_key": self._serialize_to_bytes(self.keys.publicKey, "PublicKey"),
        "mult_key": self._serialize_to_bytes(None, "MultKey"),
        "rot_key": self._serialize_to_bytes(None, "RotKey"),
    }
    
  def generate_keys(self, **kwargs):
    t0 = time.perf_counter()
    self.keys=self.context.KeyGen()
    self.context.EvalMultKeyGen(self.keys.secretKey)
    self.context.EvalSumKeyGen(self.keys.secretKey)
    self.keygen_time = time.perf_counter() - t0
    return self.keygen_time
    
  def get_private_key(self):
    ctx = self.get_context()
    ctx["secret_key"] = self._serialize_to_bytes(self.keys.secretKey, "SecretKey")
    return ctx
    
  def get_public_key(self):
    return self.keys.publicKey
    
  def encrypt(self, value, **kwargs):
    if not self.keys or not self.keys.publicKey:
        raise RuntimeError("Public key not available. Please generate keys first.")

    if isinstance(value, (int, float)):
      ptxt = self.context.MakeCKKSPackedPlaintext([value])
      return self.context.Encrypt(self.keys.publicKey, ptxt)

    # Flatten array of any dimension
    flat_value = np.array(value).flatten().tolist()
    max_slots = self.ring_dim // 2
    
    if len(flat_value) <= max_slots:
        ptxt = self.context.MakeCKKSPackedPlaintext(flat_value)
        return self.context.Encrypt(self.keys.publicKey, ptxt)
    else:
        chunks = [flat_value[i:i + max_slots] for i in range(0, len(flat_value), max_slots)]
        res = []
        for chunk in chunks:
            ptxt = self.context.MakeCKKSPackedPlaintext(chunk)
            res.append(self.context.Encrypt(self.keys.publicKey, ptxt))
        return np.array(res, dtype=object)
    
  def decrypt(self, value, **kwargs):
    if not self.keys or not self.keys.secretKey:
        raise RuntimeError("Secret key not available. Please generate keys first.")

    # `value` is the ciphertext or an array of ciphertexts
    if isinstance(value, Ciphertext): # A single ciphertext for a vector or scalar
        plaintext_result = self.context.Decrypt(self.keys.secretKey, value)
        original_length = plaintext_result.GetLength()
        decrypted_vector = plaintext_result.GetRealPackedValue()
        # If it was a scalar, return a scalar
        if original_length == 1:
            return decrypted_vector[0]
        # Return the decrypted 1D vector
        return np.array(decrypted_vector[:original_length])
    elif isinstance(value, (np.ndarray, list)):
        value = np.array(value, dtype=object)
        if value.size > 0 and isinstance(value.flatten()[0], Ciphertext):
            decrypted_chunks = []
            flat_val = value.flatten()
            for i in range(len(flat_val)):
                c = flat_val[i]
                plaintext_result = self.context.Decrypt(self.keys.secretKey, c)
                original_length = plaintext_result.GetLength()
                dec_vec = plaintext_result.GetRealPackedValue()
                decrypted_chunks.append(dec_vec[:original_length])
                flat_val[i] = None # Free C++ object
            del flat_val
            return np.concatenate(decrypted_chunks)
        return value
    else:
      raise ValueError(f"Invalid value type for decryption: {type(value)}. Must be a Ciphertext.")
    
  def enc_dot(self, term1, term2, **kwargs):
    # Assumes term1 and term2 are encrypted.
    # term1 can be a ciphertext (vector) or np.array of ciphertexts (matrix)
    # term2 can be a ciphertext (vector) or np.array of ciphertexts (matrix)
    
    is_term1_vec = isinstance(term1, Ciphertext)
    is_term2_vec = isinstance(term2, Ciphertext)

    if is_term1_vec and is_term2_vec:
        # Vector @ Vector -> scalar ciphertext
        return self.vec_dot(term1, term2)

    if not is_term1_vec and is_term2_vec:
        # Matrix @ Vector -> np.array of scalar ciphertexts
        return self.mat_vec_dot(term1, term2)

    if is_term1_vec and not is_term2_vec:
        # Vector @ Matrix -> np.array of scalar ciphertexts
        # The matrix (term2) is assumed to be transposed before encryption.
        return self.vec_mat_dot(term1, term2)

    if not is_term1_vec and not is_term2_vec:
        # Matrix @ Matrix -> 2D np.array of scalar ciphertexts
        # The second matrix (term2) is assumed to be transposed before encryption.
        return self.matrix_dot(term1, term2)

    raise ValueError(f"Invalid operand types for enc_dot: {type(term1)}, {type(term2)}")
    
  def vec_dot(self, term1, term2):
    # term1, term2 are ciphertexts of vectors
    mult_result = self.context.EvalMult(term1, term2)
    # Sum all elements of the resulting ciphertext
    return self.context.EvalSum(mult_result, self.context.GetEncodingParams().GetBatchSize())

  def mat_vec_dot(self, term1, term2):
    # term1: np.array of row ciphertexts
    # term2: ciphertext of a vector
    # Returns: np.array of scalar ciphertexts
    return np.array([self.vec_dot(row_ctx, term2) for row_ctx in term1], dtype=object)

  def vec_mat_dot(self, term1, term2):
    # term1: ciphertext of a vector
    # term2: np.array of column ciphertexts (matrix has been transposed)
    # Returns: np.array of scalar ciphertexts
    return np.array([self.vec_dot(term1, col_ctx) for col_ctx in term2], dtype=object)

  def matrix_dot(self, term1, term2):
    # term1: np.array of row ciphertexts
    # term2: np.array of column ciphertexts (transposed M2)
    # Returns: 2D np.array of scalar ciphertexts
    num_rows1 = len(term1)
    num_cols2 = len(term2)
    return np.array([[self.vec_dot(term1[i], term2[j]) 
                      for j in range(num_cols2)] 
                     for i in range(num_rows1)], dtype=object)
    
  def enc_add(self, term1, term2, **kwargs):
    # Both terms are encrypted.
    is_term1_vec = isinstance(term1, Ciphertext)
    is_term2_vec = isinstance(term2, Ciphertext)

    # Vector (Ciphertext) + Vector (Ciphertext)
    if is_term1_vec and is_term2_vec:
        return self.context.EvalAdd(term1, term2)
    
    # Matrix (np.array of Ciphertext) + Matrix (np.array of Ciphertext)
    if not is_term1_vec and not is_term2_vec:
        if term1.shape != term2.shape:
            raise ValueError("Matrices must have the same dimensions for addition.")
        # Element-wise addition of row ciphertexts
        return np.array([self.context.EvalAdd(term1[i], term2[i]) for i in range(len(term1))], dtype=object)

    # Matrix (np.array of Ciphertext) + Vector (Ciphertext) -> broadcast
    if not is_term1_vec and is_term2_vec:
        return np.array([self.context.EvalAdd(row_ctx, term2) for row_ctx in term1], dtype=object)
    
    # Vector (Ciphertext) + Matrix (np.array of Ciphertext) -> broadcast
    if is_term1_vec and not is_term2_vec:
        return np.array([self.context.EvalAdd(term1, row_ctx) for row_ctx in term2], dtype=object)

    raise ValueError(f"Unsupported types for enc_add: {type(term1)}, {type(term2)}")
    
  def enc_matmul(self, term1, term2, **kwargs):
    return self.enc_dot(term1, term2)
    
  def enc_mul(self, term1, term2, **kwargs):
    is_term1_vec = isinstance(term1, Ciphertext)
    
    if isinstance(term2, (int, float, np.number)):
        if is_term1_vec:
            # OpenFHE automatically rescales post EvalMult if using FLEXIBLEAUTO scaling technique
            return self.context.EvalMult(term1, float(term2))
        else:
            term1_flat = term1.flatten()
            res_flat = [self.context.EvalMult(c, float(term2)) if isinstance(c, Ciphertext) else c * term2 for c in term1_flat]
            return np.array(res_flat, dtype=object).reshape(term1.shape)
            
    return term1 * term2

  def serialize(self, value):
    if isinstance(value, Ciphertext):
        return self._serialize_to_bytes(value, "Ciphertext")
    elif isinstance(value, np.ndarray):
        value = np.array(value, dtype=object)
        flat_val = value.flatten()
        ser_val = []
        for i in range(len(flat_val)):
            c = flat_val[i]
            if isinstance(c, Ciphertext):
                ser_val.append(self._serialize_to_bytes(c, "Ciphertext"))
            else:
                ser_val.append(c)
            flat_val[i] = None # Release the Ciphertext object immediately
        del flat_val
        return np.array(ser_val, dtype=object).reshape(value.shape)
    return value

  def deserialize(self, value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        return self._deserialize_from_bytes(value, "Ciphertext")
    elif isinstance(value, (np.ndarray, list)):
        value = np.array(value, dtype=object)
        flat_val = value.flatten()
        deser_val = []
        for i in range(len(flat_val)):
            b = flat_val[i]
            if isinstance(b, (bytes, np.bytes_)):
                deser_val.append(self._deserialize_from_bytes(bytes(b), "Ciphertext"))
            else:
                deser_val.append(b)
            flat_val[i] = None # Release byte string memory
        del flat_val
        return np.array(deser_val, dtype=object).reshape(value.shape)
    return value