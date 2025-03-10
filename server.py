from src.core.util.data import load_datasets
from src.federated_learning.factory import FederatedLearningFactory
from src.encryption.factory import HomomorphicEncrytionFactory
from src.federated_learning.federation.strategy import FedCustom
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents
from src.federated_learning.federation.training_fit import on_fit_config
from src.federated_learning.federation.evaluate_fl import evaluate_fl
from src.core.util.common import get_parameters2
from src.core.util.mapper import (
  ndarrays_to_parameters_custom,
  parameters_to_ndarrays,
  ndarray_to_bytes,
  bytes_to_ndarray,
)
from src.core.util.load_config import LoadConfig
from src.core.util.common import read_file, write_file, load_file
from src.federated_learning.federation.weighted_average import weighted_average
from example.models.custom_model import Net
# from example.models.quantum_model import Net
import tenseal as ts
import os
import flwr as fl
import torch

os.environ["GRPC_MAX_RECEIVE_MESSAGE_LENGTH"] = str(1024 * 1024 * 1024)
os.environ["GRPC_MAX_SEND_MESSAGE_LENGTH"] = str(1024 * 1024 * 1024) 

def server_fn(context: Context):
    # Register the custom adapters
    fl.common.parameter.ndarrays_to_parameters = ndarrays_to_parameters
    fl.common.parameter.paramaters_to_ndarrays = parameters_to_ndarrays
    fl.common.parameter.ndarray_to_bytes = ndarray_to_bytes
    fl.common.parameter.bytes_to_ndarray = bytes_to_ndarray
  
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]
    min_fit_clients = context.run_config["min-fit-clients"]
    min_eval_clients = context.run_config["min-evaluate-clients"]
    min_avail_clients = context.run_config["min-available-clients"]
    number_clients = context.run_config["number-clients"]
    
    # Loading extra config
    config = LoadConfig("pyproject.toml")
    nn = config.get_nn_config()
    he = config.get_he_config()
    test = config.get_test_config()
    stats = config.get_stats_config()
    quantum = config.get_quantum_config()
    
    # Set the device
    device = torch.device(server_device)
    
    # Load the test dataset
    _, _, testloader = load_datasets(1, test["batch_size"], test["resize"], test["seed"], test["num_workers"], 
                                     test["splitter"], test["dataset"], test["data_path"], test["validation_data"], 
                                     test["normalization"]["mean"], test["normalization"]["std"])
    context_server = None
    
    if he["enable"]:
      
      # Setup HE
      he_backend = HomomorphicEncrytionFactory.get_backend(library=he["library"], schema=he["schema"])
      he_backend.set_poly_modulus_degree(he["poly_modulus_degree"])
      he_backend.set_global_scale(he["global_scale"])
      he_backend.set_coef_mod_bit(he["coef_mod_bit"])
    
      if not os.path.exists(he["context"]["client"]):
        # Save HE context
        private_key = he_backend.get_private_key()
        write_file(he["context"]["client"], {"contexte": private_key})
        write_file(he["context"]["server"], {"contexte": he_backend.get_context()}) 
        
        write_file(he["key"]["client"], {"contexte": private_key})
        write_file(he["key"]["server"], {"contexte": he_backend.get_context()}) 
         
        _, context_client = read_file(he["context"]["client"])
        _, context_server = read_file(he["context"]["server"])
        
        # raise Exception("Public key not found")
      else:
        _, context_client = read_file(he["context"]["client"])
        _, context_server = read_file(he["context"]["server"])
        
        # he_backend.set_context(server_context)
    
    # Load the central model
    central = Net(num_classes=len(test["classes"]), 
                  num_layers=quantum["num_layers"], 
                  num_qubits=quantum["num_qubits"]).to(device)

    # Loading federated client
    fl_server = FederatedLearningFactory.get_backend('FLWR-SERVER')
    
    # Set the strategy
    strategy = fl_server.server(func=FedCustom, fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients if min_eval_clients else number_clients // 2,
        min_available_clients=min_avail_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters_custom(get_parameters2(central)),
        evaluate_fn=None if he["enable"] else evaluate_fl,
        on_fit_config_fn=on_fit_config(epoch=nn["max_epochs"], batch_size=nn["batch_size"], lr=nn["learning_rate"]),
        context_server= he_backend.set_context(context_server) if he["enable"] else None,
        lr=nn["learning_rate"],
        model_save=nn["model_save"],  
        central_model=central,
        test_loader=testloader,
        device=device,
        checkpoint=nn["checkpoint"],
        path_crypted=he["key"]["server_crypted"],
        )

    #config = ServerConfig(num_rounds=num_rounds)
    config = fl_server.server_config(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)