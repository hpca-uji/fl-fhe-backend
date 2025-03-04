from .backend.flower.client import Client as FlowerClient
from .backend.flower.server import FlowerServer
class FederatedLearningFactory:
  _backend = {}
  
  @classmethod
  def register_backend(cls, library, backend_class):
    cls._backend[(library)] = backend_class
  
  @classmethod
  def get_backend(cls, library):
    backend_cls = cls._backend.get((library))
    if not backend_cls:
      raise ValueError(f"Unsupported library '{library}'")
    return backend_cls
  
FederatedLearningFactory.register_backend(library='FLWR-CLIENT', backend_class=FlowerClient)
FederatedLearningFactory.register_backend(library='FLWR-SERVER', backend_class=FlowerServer)