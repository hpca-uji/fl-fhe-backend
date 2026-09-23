import os
import sys
import time
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from src.core.util.load_config import LoadConfig
from src.core.util.common import check_directory


def calculate_weights_size(parameters: Union[List[Any], Dict[str, Any], Any]) -> Tuple[int, float, float]:
    """
    Calculate the total size of weights / parameters in bytes, KB, and MB.
    Supports bytes, numpy arrays (numeric or object containing ciphertexts/bytes),
    PyTorch tensors, and FHE objects with serialize/to_bytes methods.
    
    :param parameters: List or dictionary of weights/parameters
    :return: Tuple of (size_bytes, size_kb, size_mb)
    """
    total_bytes = 0
    
    if parameters is None:
        return 0, 0.0, 0.0
        
    if isinstance(parameters, dict):
        items = list(parameters.values())
    elif isinstance(parameters, (list, tuple)):
        items = list(parameters)
    else:
        items = [parameters]
        
    for item in items:
        if item is None:
            continue
        elif isinstance(item, (bytes, bytearray, memoryview)):
            total_bytes += len(item)
        elif isinstance(item, np.ndarray):
            if item.dtype == object:
                for sub_item in item.flatten():
                    if sub_item is None:
                        continue
                    elif isinstance(sub_item, (bytes, bytearray, memoryview)):
                        total_bytes += len(sub_item)
                    elif hasattr(sub_item, "serialize"):
                        try:
                            ser = sub_item.serialize()
                            total_bytes += len(ser) if isinstance(ser, (bytes, bytearray, str)) else sys.getsizeof(ser)
                        except Exception:
                            total_bytes += sys.getsizeof(sub_item)
                    elif hasattr(sub_item, "to_bytes"):
                        try:
                            ser = sub_item.to_bytes()
                            total_bytes += len(ser) if isinstance(ser, (bytes, bytearray, str)) else sys.getsizeof(ser)
                        except Exception:
                            total_bytes += sys.getsizeof(sub_item)
                    else:
                        total_bytes += sys.getsizeof(sub_item)
            else:
                total_bytes += item.nbytes
        elif isinstance(item, torch.Tensor):
            total_bytes += item.element_size() * item.nelement()
        elif hasattr(item, "serialize"):
            try:
                ser = item.serialize()
                total_bytes += len(ser) if isinstance(ser, (bytes, bytearray, str)) else sys.getsizeof(ser)
            except Exception:
                total_bytes += sys.getsizeof(item)
        elif hasattr(item, "to_bytes"):
            try:
                ser = item.to_bytes()
                total_bytes += len(ser) if isinstance(ser, (bytes, bytearray, str)) else sys.getsizeof(ser)
            except Exception:
                total_bytes += sys.getsizeof(item)
        elif hasattr(item, "tensors"):
            # Flower Parameters object
            for t in item.tensors:
                total_bytes += len(t)
        else:
            total_bytes += sys.getsizeof(item)
            
    size_kb = total_bytes / 1024.0
    size_mb = total_bytes / (1024.0 * 1024.0)
    return total_bytes, size_kb, size_mb


def log_keygen_metrics(
    component: str,
    keygen_time_sec: float,
    fhe_lib: Optional[str] = None,
    poly_modulus_degree: Optional[int] = None,
    results_dir: str = "results"
) -> str:
    """
    Log key generation time to CSV.
    
    :param component: 'client' or 'server'
    :param keygen_time_sec: Time in seconds to generate keys
    :param fhe_lib: FHE library used (e.g., OPENFHE, TENSEAL, PYFHEL)
    :param poly_modulus_degree: Polynomial modulus degree
    :param results_dir: Base directory for results
    :return: Path of the written CSV file
    """
    try:
        config = LoadConfig("pyproject.toml")
        he_cfg = config.get_he_config()
        if fhe_lib is None:
            fhe_lib = he_cfg.get("library", "NONE")
        if poly_modulus_degree is None:
            poly_modulus_degree = he_cfg.get("poly_modulus_degree", 0)
    except Exception:
        fhe_lib = fhe_lib or "UNKNOWN"
        poly_modulus_degree = poly_modulus_degree or 0

    device_name = socket.gethostname()
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "fhe_keygen_stats.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, "a") as f:
        if not file_exists:
            f.write("date,component,fhe_lib,poly_modulus_degree,device_name,keygen_time_sec\n")
        f.write(f"{current_date},{component},{fhe_lib},{poly_modulus_degree},{device_name},{keygen_time_sec:.6f}\n")
        
    return csv_path


def log_client_fhe_metrics(
    client_id: Union[str, int],
    server_round: int,
    decryption_time_sec: float,
    encryption_time_sec: float,
    weights_size_bytes: int,
    keygen_time_sec: float = 0.0,
    model: Optional[str] = None,
    fhe_lib: Optional[str] = None,
    poly_modulus_degree: Optional[int] = None,
    results_dir: str = "results"
) -> str:
    """
    Log client-side FHE timing metrics and transmitted weights size to CSV.
    
    :param client_id: Identifier of the client partition
    :param server_round: Federated learning round number
    :param decryption_time_sec: Time in seconds to decrypt global model weights
    :param encryption_time_sec: Time in seconds to encrypt local model weights
    :param weights_size_bytes: Size in bytes of weights passed to server
    :param keygen_time_sec: Key generation time (if performed on client)
    :param model: Neural network model architecture
    :param fhe_lib: FHE library name
    :param poly_modulus_degree: Polynomial modulus degree
    :param results_dir: Base directory for results
    :return: Path to the written client CSV file
    """
    try:
        config = LoadConfig("pyproject.toml")
        nn_cfg = config.get_nn_config()
        he_cfg = config.get_he_config()
        if model is None:
            model = nn_cfg.get("model", "UNKNOWN")
        if fhe_lib is None:
            fhe_lib = he_cfg.get("library", "NONE") if he_cfg.get("enable", False) else "NONE"
        if poly_modulus_degree is None:
            poly_modulus_degree = he_cfg.get("poly_modulus_degree", 0) if he_cfg.get("enable", False) else 0
    except Exception:
        model = model or "UNKNOWN"
        fhe_lib = fhe_lib or "NONE"
        poly_modulus_degree = poly_modulus_degree or 0

    device_name = socket.gethostname()
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    weights_size_kb = weights_size_bytes / 1024.0
    weights_size_mb = weights_size_bytes / (1024.0 * 1024.0)

    client_results_dir = os.path.join(results_dir, f"client_{client_id}")
    os.makedirs(client_results_dir, exist_ok=True)
    
    client_csv_path = os.path.join(client_results_dir, f"fhe_metrics_{device_name}.csv")
    file_exists = os.path.isfile(client_csv_path)
    
    header = "date,client_id,round,model,fhe_lib,poly_modulus_degree,device_name,keygen_time_sec,decryption_time_sec,encryption_time_sec,weights_size_bytes,weights_size_kb,weights_size_mb\n"
    row = f"{current_date},{client_id},{server_round},{model},{fhe_lib},{poly_modulus_degree},{device_name},{keygen_time_sec:.6f},{decryption_time_sec:.6f},{encryption_time_sec:.6f},{weights_size_bytes},{weights_size_kb:.4f},{weights_size_mb:.6f}\n"

    with open(client_csv_path, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(row)

    # Also record to central summary CSV
    summary_csv = os.path.join(results_dir, "fhe_client_metrics_summary.csv")
    summary_exists = os.path.isfile(summary_csv)
    with open(summary_csv, "a") as f:
        if not summary_exists:
            f.write(header)
        f.write(row)

    return client_csv_path


def log_server_fhe_metrics(
    server_round: int,
    homomorphic_addition_time_sec: float,
    homomorphic_mult_time_sec: float = 0.0,
    total_aggregation_time_sec: float = 0.0,
    received_weights_size_bytes: int = 0,
    num_clients: int = 0,
    keygen_time_sec: float = 0.0,
    model: Optional[str] = None,
    fhe_lib: Optional[str] = None,
    poly_modulus_degree: Optional[int] = None,
    results_dir: str = "results"
) -> str:
    """
    Log server-side FHE timing metrics (homomorphic addition, aggregation) and received weights size to CSV.
    
    :param server_round: Federated learning round number
    :param homomorphic_addition_time_sec: Time in seconds for homomorphic additions
    :param homomorphic_mult_time_sec: Time in seconds for homomorphic scalar multiplications
    :param total_aggregation_time_sec: Total round aggregation time
    :param received_weights_size_bytes: Total size in bytes of weights received from clients
    :param num_clients: Number of clients that participated in this round
    :param keygen_time_sec: Key generation time (if performed on server)
    :param model: Neural network model architecture
    :param fhe_lib: FHE library name
    :param poly_modulus_degree: Polynomial modulus degree
    :param results_dir: Base directory for results
    :return: Path to the written server CSV file
    """
    try:
        config = LoadConfig("pyproject.toml")
        nn_cfg = config.get_nn_config()
        he_cfg = config.get_he_config()
        if model is None:
            model = nn_cfg.get("model", "UNKNOWN")
        if fhe_lib is None:
            fhe_lib = he_cfg.get("library", "NONE") if he_cfg.get("enable", False) else "NONE"
        if poly_modulus_degree is None:
            poly_modulus_degree = he_cfg.get("poly_modulus_degree", 0) if he_cfg.get("enable", False) else 0
    except Exception:
        model = model or "UNKNOWN"
        fhe_lib = fhe_lib or "NONE"
        poly_modulus_degree = poly_modulus_degree or 0

    device_name = socket.gethostname()
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    received_weights_size_kb = received_weights_size_bytes / 1024.0
    received_weights_size_mb = received_weights_size_bytes / (1024.0 * 1024.0)
    avg_client_weights_size_mb = (received_weights_size_mb / num_clients) if num_clients > 0 else 0.0

    server_results_dir = os.path.join(results_dir, "server")
    os.makedirs(server_results_dir, exist_ok=True)

    server_csv_path = os.path.join(server_results_dir, f"fhe_metrics_{device_name}.csv")
    file_exists = os.path.isfile(server_csv_path)

    header = "date,round,model,fhe_lib,poly_modulus_degree,num_clients,device_name,keygen_time_sec,homomorphic_addition_time_sec,homomorphic_mult_time_sec,total_aggregation_time_sec,received_weights_size_bytes,received_weights_size_kb,received_weights_size_mb,avg_client_weights_size_mb\n"
    row = f"{current_date},{server_round},{model},{fhe_lib},{poly_modulus_degree},{num_clients},{device_name},{keygen_time_sec:.6f},{homomorphic_addition_time_sec:.6f},{homomorphic_mult_time_sec:.6f},{total_aggregation_time_sec:.6f},{received_weights_size_bytes},{received_weights_size_kb:.4f},{received_weights_size_mb:.6f},{avg_client_weights_size_mb:.6f}\n"

    with open(server_csv_path, "a") as f:
        if not file_exists:
            f.write(header)
        f.write(row)

    # Also record to central summary CSV
    summary_csv = os.path.join(results_dir, "fhe_server_metrics_summary.csv")
    summary_exists = os.path.isfile(summary_csv)
    with open(summary_csv, "a") as f:
        if not summary_exists:
            f.write(header)
        f.write(row)

    return server_csv_path
