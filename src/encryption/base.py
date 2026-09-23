from abc import ABC, abstractmethod

class EncryptionBase(ABC):
    def __init__(self):
        self.keygen_time = 0.0

    def get_keygen_time(self) -> float:
        return self.keygen_time

    @abstractmethod
    def get_context(self):
        pass
    
    @abstractmethod
    def generate_keys(self, **kwargs):
        pass
    
    @abstractmethod
    def get_private_key(self):
        pass
    
    @abstractmethod
    def get_public_key(self):
        pass
    
    @abstractmethod
    def encrypt(self, value, **kwargs):
        pass
    
    @abstractmethod
    def decrypt(self, value, **kwargs):
        pass
    
    @abstractmethod
    def enc_dot(self, term1, term2, **kwargs):
        pass
    
    @abstractmethod
    def enc_add(self, term1, term2, **kwargs):
        pass
    
    @abstractmethod
    def enc_matmul(self, term1, term2, **kwargs):
        pass
    
    @abstractmethod
    def enc_mul(self, term1, term2, **kwargs):
        pass
    