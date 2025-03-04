from abc import ABC, abstractmethod

class FLServerBase(ABC):

  @abstractmethod
  def server(self, func, **kwargs):
    pass