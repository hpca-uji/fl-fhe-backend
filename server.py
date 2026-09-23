import os
import time
import numpy as np
import torch
import flwr as fl
try:
    import tenseal as ts
except ImportError:
    ts = None
import torchvision.models as models
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents

# Existing project imports
from src.core.util.common import get_parameters2, read_file, write_file
from src.core.util.load_config import LoadConfig
from src.core.util.mapper import *
from src.federated_learning.factory import FederatedLearningFactory
from src.encryption.factory import HomomorphicEncrytionFactory
from src.federated_learning.federation.strategy import FedCustom
from src.federated_learning.federation.training_fit import on_fit_config
from src.federated_learning.federation.weighted_average import weighted_average
from src.federated_learning.federation.evaluate_fl import evaluate_fl
from src.core.util.fhe_metrics import log_keygen_metrics

import torchxrayvision as xrv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Force large message limits for HE ciphertexts
os.environ["GRPC_MAX_RECEIVE_MESSAGE_LENGTH"] = str(2147483648)
os.environ["GRPC_MAX_SEND_MESSAGE_LENGTH"] = str(2147483648)

class XRVAdapter(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes=25): # Changed from 13 to 25
        self.dataset = dataset
        self.num_classes = num_classes
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        item = self.dataset[i]
        lab = item["lab"]
        # Pad with zeros if dataset has fewer classes than required
        if len(lab) < self.num_classes:
            lab = np.pad(lab, (0, self.num_classes - len(lab)))
        return item["img"], lab[:self.num_classes]

def server_fn(context: Context):
    # Config extraction
    num_rounds = context.run_config["num-server-rounds"]
    min_eval_clients = context.run_config["min-evaluate-clients"]
    server_device = context.run_config["server-device"]
    number_clients = context.run_config["number-clients"]
    
    # Set the device
    device = torch.device(server_device)
    
    config = LoadConfig("pyproject.toml")
    nn = config.get_nn_config()
    he = config.get_he_config()
    test_cfg = config.get_test_config()
    test = config.get_test_config()
    
    partition_id = 0 #int(context.node_config["partition-id"])
    
    # Server should ideally evaluate on a global test set. 
    # Using client_0.csv with Non-IID data will lead to biased evaluation.
    client_csv = "data/splits/test.csv"
    # if not os.path.exists(client_csv):
    #     client_csv = f"data/splits/client_{partition_id}.csv"
    img_path = "data/images/"

    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])

    raw_ds = xrv.datasets.COVID19_Dataset(
        imgpath=img_path,
        csvpath=client_csv,
        transform=transform # Use the transform with np.squeeze discussed earlier
    )
    
    num_labels = raw_ds.labels.shape[1] 
    print(f"Client {partition_id} detected {num_labels} classes.")

    # Wrap and Load
    full_ds = XRVAdapter(raw_ds, num_classes=len(test["classes"]))
    
    testloader = DataLoader(full_ds, batch_size=32, shuffle=True)
    
    # Load model based on config
    num_classes = len(test["classes"])
    if nn["model"] == "SQUEEZE":
        central = models.squeezenet1_1(weights=None)
        central.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        central.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif nn["model"] == "MOBILENETV3":
        central = models.mobilenet_v3_small(weights=None)
        central.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        central.classifier[3] = torch.nn.Linear(central.classifier[3].in_features, num_classes)
    elif nn["model"] == "SHUFFLENET":
        central = models.shufflenet_v2_x0_5(weights=None)
        central.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        central.fc = torch.nn.Linear(central.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {nn['model']}")

    central.num_classes = num_classes
    central.classes = test["classes"]
    central = central.to(device)

    context_server = None
    he_backend = None

    if he["enable"]:
        he_backend = HomomorphicEncrytionFactory.get_backend(library=he["library"], schema=he["schema"])
        
        he_backend.set_poly_modulus_degree(he.get("poly_modulus_degree", 4096))
        he_backend.set_global_scale(he.get("global_scale", 2**20))
        he_backend.set_coef_mod_bit(he.get("coef_mod_bit", [40, 20, 40]))
        he_backend.create_context()
    
        force_regen = False
        if os.path.exists(he["context"]["client"]):
            try:
                query, client_context = read_file(he["context"]["client"])
                if he["library"] != "TENSEAL" and isinstance(client_context, bytes):
                    force_regen = True
                elif query.get("poly_modulus_degree") != he.get("poly_modulus_degree", 4096):
                    force_regen = True
            except Exception:
                force_regen = True

        if not os.path.exists(he["context"]["client"]) or force_regen:
            os.makedirs(os.path.dirname(he["context"]["client"]), exist_ok=True)
            
            t0_keygen = time.perf_counter()
            he_backend.generate_keys()
            keygen_time = getattr(he_backend, "keygen_time", time.perf_counter() - t0_keygen)
            he_backend.keygen_time = keygen_time
            log_keygen_metrics(
                component="server",
                keygen_time_sec=keygen_time,
                fhe_lib=he["library"],
                poly_modulus_degree=he.get("poly_modulus_degree", 4096)
            )
            private_key = he_backend.get_private_key()
            write_file(he["context"]["client"], {
                "context": private_key,
                "poly_modulus_degree": he.get("poly_modulus_degree", 4096),
                "keygen_time": keygen_time
            })
            write_file(he["context"]["server"], {
                "context": he_backend.get_context(),
                "keygen_time": keygen_time
            }) 
        
        query_srv, context_server_raw = read_file(he["context"]["server"])
        he_backend.set_context(context_server_raw)
        he_backend.keygen_time = query_srv.get("keygen_time", 0.0) if isinstance(query_srv, dict) else 0.0
        context_server = he_backend

    fl_server = FederatedLearningFactory.get_backend('FLWR-SERVER')
    
    strategy = fl_server.server(
        func=FedCustom, 
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_fit_clients=context.run_config["min-fit-clients"],
        min_evaluate_clients=min_eval_clients if min_eval_clients else number_clients // 2,
        min_available_clients=context.run_config["min-available-clients"],
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters_custom(get_parameters2(central)),
        on_fit_config_fn=on_fit_config(epoch=nn["max_epochs"], batch_size=nn["batch_size"], lr=nn["learning_rate"]),
        # context_server= he_backend.set_context(context_server) if he["enable"] else None,
        evaluate_fn=None if he["enable"] else evaluate_fl,
        # on_fit_config_fn=on_fit_config(
        #     epoch=nn["max_epochs"], 
        #     batch_size=1, # Match the client's ultra-low batch size
        #     lr=nn["learning_rate"]
        # ),
        context_server=context_server,
        lr=nn["learning_rate"],
        model_save=nn["model_save"] + "_" + nn["model"] + ".pkl",  
        central_model=central,
        test_loader=testloader,
        device=device,
        checkpoint=nn["checkpoint"] + "_" + nn["model"] + ".pkl",
        path_crypted=he["key"]["server_crypted"] if he["enable"] else None,
    )

    server_config = fl_server.server_config(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)