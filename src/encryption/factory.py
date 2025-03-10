from .backends.tenseal.ckks import TensealCKKS
from .backends.openfhe.ckks import OpenFHECKKS
from .backends.pyfhel.ckks import PyFHELCKKS


class HomomorphicEncrytionFactory:
    _backend = {}
    
    @classmethod
    def register_backend(cls, library, schema, backend_class):
        cls._backend[(library, schema)] = backend_class
        
    @classmethod
    def get_backend(cls, library='TENSEAL', schema='CKKS'):
        backend_cls = cls._backend.get((library, schema))
        if not backend_cls:
            raise ValueError(f"Unsupported library '{library}' or schema '{schema}'")
        return backend_cls()
    
# Register supported libraries and schemas
HomomorphicEncrytionFactory.register_backend(library='TENSEAL', schema='CKKS', backend_class=TensealCKKS)
HomomorphicEncrytionFactory.register_backend(library='OPENFHE', schema='CKKS', backend_class=OpenFHECKKS)
HomomorphicEncrytionFactory.register_backend(library='PYFHEL', schema='CKKS', backend_class=PyFHELCKKS)