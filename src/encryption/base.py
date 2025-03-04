from abc import ABC, abstractmethod

class EncryptionBase(ABC):
    
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
    