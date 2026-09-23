import os
import gc
import time
import torch
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as transforms
from flwr.client import ClientApp
from flwr.common import Context
import torchxrayvision as xrv

# Existing project imports
from src.core.util.common import *
from src.core.util.mapper import *
from src.core.util.load_config import LoadConfig
from src.federated_learning.factory import FederatedLearningFactory
from src.nn.layers.layer_util import deserialized_layer
from src.encryption.factory import HomomorphicEncrytionFactory
from src.core.util.fhe_metrics import log_keygen_metrics

os.environ["GRPC_MAX_RECEIVE_MESSAGE_LENGTH"] = str(2147483648)
os.environ["GRPC_MAX_SEND_MESSAGE_LENGTH"] = str(2147483648)

class XRVAdapter(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes=25): 
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

def client_fn(context: Context):
    gc.collect()
    
    # Load Configs
    partition_id = int(context.node_config["partition-id"])
    num_partitions = context.node_config["num-partitions"]
    config = LoadConfig("pyproject.toml")
    nn = config.get_nn_config()
    he = config.get_he_config()
    test_cfg = config.get_test_config()
    test = config.get_test_config()
    stats = config.get_stats_config()
    fl_config = config.get_app_config()
    
    device = torch.device("cpu")

    # partition_id = int(fl_config["client"])
    print(f"------------------- Client {partition_id} --------------------\n")
    client_csv = f"data/splits/client_{partition_id}.csv"
    img_path = "data/images/"

    # Replaced ImageNet normalization with XRV native transformations
    transform = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224) 
    ])

    raw_ds = xrv.datasets.COVID19_Dataset(
        imgpath=img_path,
        csvpath=client_csv,
        transform=transform 
    )
    
    num_labels = raw_ds.labels.shape[1] 
    print(f"Client {partition_id} detected {num_labels} classes.")

    # Wrap and Load
    num_classes = len(test["classes"])
    full_ds = XRVAdapter(raw_ds, num_classes=num_classes)
    
    trainloader = DataLoader(full_ds, batch_size=32, shuffle=True)
    valloader = DataLoader(full_ds, batch_size=32, shuffle=False)

    # Load model based on config, with no pre-trained weights (train from scratch)
    if nn["model"] == "SQUEEZE":
        net = models.squeezenet1_1(weights=None)
        # Patch for 1-channel grayscale input (instead of 3-channel RGB)
        net.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # Patch classifier for custom number of classes
        net.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    elif nn["model"] == "MOBILENETV3":
        net = models.mobilenet_v3_small(weights=None)
        # Patch for 1-channel grayscale input
        net.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        # Patch classifier for custom number of classes
        net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, num_classes)
    elif nn["model"] == "SHUFFLENET":
        net = models.shufflenet_v2_x0_5(weights=None)
        net.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model architecture: {nn['model']}")
    net.num_classes = num_classes
    net = net.to(device)

    context_client = None
    
    if he["enable"]:
        # Setup HE
        he_backend = HomomorphicEncrytionFactory.get_backend(library=he["library"], schema=he["schema"])
        he_backend.set_poly_modulus_degree(he["poly_modulus_degree"])
        he_backend.set_global_scale(he["global_scale"])
        he_backend.set_coef_mod_bit(he["coef_mod_bit"])
        he_backend.create_context()
    
        print("Run with homomorphic encryption")
        force_regen = False
        if os.path.exists(he["context"]["client"]):
            with open(he["context"]["client"], 'rb') as f:
                query = pickle.load(f)
            if query.get("poly_modulus_degree") != he["poly_modulus_degree"]:
                force_regen = True
            else:
                if he["library"] == "TENSEAL":
                    he_backend.set_context(query["context"])
                else:
                    if isinstance(query["context"], bytes):
                        force_regen = True
                    else:
                        he_backend.set_private_key(query["context"])
                he_backend.keygen_time = query.get("keygen_time", 0.0)
                context_client = he_backend
            
        if not os.path.exists(he["context"]["client"]) or force_regen:
            t0_keygen = time.perf_counter()
            he_backend.generate_keys()
            keygen_time = getattr(he_backend, "keygen_time", time.perf_counter() - t0_keygen)
            he_backend.keygen_time = keygen_time
            log_keygen_metrics(
                component=f"client_{partition_id}",
                keygen_time_sec=keygen_time,
                fhe_lib=he["library"],
                poly_modulus_degree=he["poly_modulus_degree"]
            )
            with open(he["context"]["client"], 'wb') as f:
                encode = pickle.dumps({
                    "context": he_backend.get_private_key(),
                    "poly_modulus_degree": he["poly_modulus_degree"],
                    "keygen_time": keygen_time
                })
                f.write(encode)
            context_client = he_backend
    else:
        print("Run WITHOUT homomorphic encryption")
        
    model_save_path = nn["model_save"] + "_" + nn["model"] + ".pkl"

    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location="cpu", weights_only=False)['model_state_dict']
        
        current_model_dict = net.state_dict()
        checkpoint = {k: v for k, v in checkpoint.items() 
                    if k in current_model_dict and v.size() == current_model_dict[k].size()}
        
        net.load_state_dict(checkpoint, strict=False)
        print("Loaded compatible layers from checkpoint")

    fl_client = FederatedLearningFactory.get_backend('FLWR-CLIENT')
    cl = fl_client.client(
        cid=partition_id, 
        net=net, 
        trainloader=trainloader, 
        valloader=valloader, 
        device=device, 
        batch_size=32,
        save_results=stats["enable"], 
        plot_path=stats["plots"], 
        roc_path=stats["plots"], 
        yaml_path=stats["reports"],
        he_enable=he["enable"], 
        classes=test["classes"], 
        context_client=context_client
    )
    
    gc.collect()
    return cl.to_client()

app = ClientApp(client_fn)
