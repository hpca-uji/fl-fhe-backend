
from src.core.util.mapper import parameters_to_ndarrays_custom
from src.core.util.common import *
import numpy as np
from typing import List, OrderedDict
import torch
from src.core.util.common import check_directory

def aggregate(server_round, aggregated_parameters, central_model, checkpoint=None,
              context_client=None, context_server=None, server_path=None):
    if aggregated_parameters is not None:
        print(f"Saving round {server_round} aggregated_parameters...")
        aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays_custom(aggregated_parameters, context_server)
        if context_client:
            server_response = {"contexte": context_server.serialize()}
            for i, key in enumerate(central_model.state_dict().keys()):
                try:
                    server_response[key] = aggregated_ndarrays[i].serialize()
                except:
                    server_response[key] = aggregated_ndarrays[i]
            write_file(server_path, server_response)
        else:
            params_dict = zip(central_model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            central_model.load_state_dict(state_dict, strict=True)
            if checkpoint:
                print(checkpoint)
                check_directory(checkpoint)
                torch.save({
                    'model_state_dict': central_model.state_dict(),
                }, checkpoint)