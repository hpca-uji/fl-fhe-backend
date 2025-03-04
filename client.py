import flwr as fl
from src.core.util.common import *
from src.nn.engine.test import test
from src.nn.engine.train import train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from src.core.util.data import load_datasets
from example.models.custom_model import Net
import tomli
import tenseal as ts
from src.nn.layers.layer_util import deserialized_layer
from src.core.util.common import read_file, check_directory
from src.encryption.factory import HomomorphicEncrytionFactory
from datetime import datetime
from src.core.util.load_config import LoadConfig
from src.core.util.common import choice_device
from random import randrange, randint
from src.federated_learning.factory import FederatedLearningFactory

os.environ["GRPC_MAX_RECEIVE_MESSAGE_LENGTH"] = str(1024 * 1024 * 1024)
os.environ["GRPC_MAX_SEND_MESSAGE_LENGTH"] = str(1024 * 1024 * 1024) 


def client_fn(context: Context):
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    server_device = context.run_config["server-device"]
    
    # Loading extra config
    config = LoadConfig("pyproject.toml")
    nn = config.get_nn_config()
    he = config.get_he_config()
    test = config.get_test_config()
    stats = config.get_stats_config()
    
    device = torch.device(server_device)
    
    # Loading Data
    trainloader, valloader, testloader = load_datasets(num_partitions, test["batch_size"], test["resize"], 
                                                       test["seed"], test["num_workers"], test["splitter"],
                                                       test["dataset"], test["data_path"], test["validation_data"], 
                                                       test["normalization"]["mean"], test["normalization"]["std"])
    
    context_client = None
    net = Net(num_classes=len(test["classes"])).to(device)

    if he["enable"]:
        # Setup HE
        he_backend = HomomorphicEncrytionFactory.get_backend(library=he["library"], schema=he["schema"])
        he_backend.set_poly_modulus_degree(he["poly_modulus_degree"])
        he_backend.set_global_scale(he["global_scale"])
        he_backend.set_coef_mod_bit(he["coef_mod_bit"])
        he_context = he_backend.get_context()
    
        print("Run with homomorphic encryption")
        if os.path.exists(he["context"]["client"]):
            with open(he["context"]["client"], 'rb') as f:
                query = pickle.load(f)
            context_client = ts.context_from(query["contexte"])
        else:
            with open(he["context"]["client"], 'wb') as f:
                encode = pickle.dumps({"contexte": he_context.serialize(save_secret_key = True)})
                f.write(encode)
        secret_key = context_client.secret_key()
    else:
        print("Run WITHOUT homomorphic encryption")

    if os.path.exists(nn["model_save"]):
        checkpoint = torch.load(nn["model_save"], map_location=server_device)['model_state_dict']
        if he["enable"]:
            server_query, server_context = read_file(he["context"]["server"])
            server_context = ts.context_from(server_context)
            for name in checkpoint:
                checkpoint[name] = torch.tensor(
                    deserialized_layer(name, server_query[name], server_context).decrypt(secret_key)
                )
        net.load_state_dict(checkpoint)
    random_num = 3
    
    # Loading federated client
    fl_client = FederatedLearningFactory.get_backend('FLWR-CLIENT')
    cl = fl_client.client(cid=1, net=net, trainloader=trainloader[1], valloader=valloader[1], 
                            device=device, batch_size=nn["batch_size"], save_results=stats["enable"], 
                            plot_path=stats["plots"], roc_path=stats["plots"], yaml_path=stats["reports"],
                            he_enable=he["enable"], classes=test["classes"], context_client=context_client)
    return cl.to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)