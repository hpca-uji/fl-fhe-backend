from typing import Optional
from src.federated_learning.server_base import FLServerBase
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

class FlowerServer(FLServerBase):
  def server(func, **kwargs):
    return func(**kwargs)
    
  def server_config(num_rounds: int = 1, round_timeout: Optional[float] = None):
    return ServerConfig(num_rounds=num_rounds, round_timeout=round_timeout)
  
  def server_app_components(self):
    pass