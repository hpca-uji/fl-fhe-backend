from abc import ABC, abstractmethod

class FLClientBase(ABC):

  @abstractmethod
  def client(self, cid, trainloader, valloader, 
                    device, batch_size, save_results, plot_path, 
                    roc_path, yaml_path, he, classes, context_client):
    pass