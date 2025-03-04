from typing import Dict, Optional, Tuple, Union
from src.core.typing.typing import NDArray, Scalar
from src.nn.engine.test import test
import torch
from src.core.util.common import set_parameters

def evaluate_fl(server_round: int, parameters: NDArray, 
                config: Dict[str, Scalar], 
              testloader, device, model_init) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    set_parameters(model_init, parameters)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    loss, accuracy, y_pred, y_true, y_proba = test(model=model_init, dataloader=testloader, loss_fn=loss_fn,
                                                          device=device)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}