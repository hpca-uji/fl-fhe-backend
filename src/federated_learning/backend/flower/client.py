from src.federated_learning.client_base import FLClientBase
from .common.flower_client import FlowerClient
from torch.utils.data import DataLoader
import torch


class Client(FLClientBase):
  def __init__(self):
    super().__init__()
    
  def client(cid=None, net=None, trainloader=None, valloader=None, 
                  device=None, batch_size=16, save_results=None, 
                  plot_path=None, roc_path=None, yaml_path=None, he_enable=False, 
                  classes=None, context_client=None):
    return FlowerClient(cid, net, trainloader, valloader, device, 
                        batch_size, save_results, plot_path, roc_path, 
                        yaml_path, he_enable, classes, context_client)