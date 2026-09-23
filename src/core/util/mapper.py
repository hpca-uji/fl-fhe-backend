import numpy as np
import torch
import pickle
from io import BytesIO
from typing import cast
from flwr.common.typing import NDArrays, NDArray, Parameters
import flwr as fl
import flwr.client.numpy_client

def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    return ndarrays_to_parameters_custom(ndarrays)

def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    return parameters_to_ndarrays_custom(parameters)

def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    return ndarray_to_bytes_custom(ndarray)

def bytes_to_ndarray(tensor: bytes) -> NDArray:
    return bytes_to_ndarray_custom(tensor)

def ndarray_to_bytes_custom(ndarray: NDArray) -> bytes:
    if isinstance(ndarray, bytes):
        return ndarray
    if hasattr(ndarray, "serialize"):
        return ndarray.serialize()
    if hasattr(ndarray, "to_bytes"):
        return ndarray.to_bytes()

    bytes_io = BytesIO()
    np.save(bytes_io, 
            ndarray.cpu().detach().numpy() if isinstance(ndarray, torch.Tensor) else ndarray, 
            allow_pickle=True
            )
    return bytes_io.getvalue()

def bytes_to_ndarray_custom(tensor: bytes) -> NDArray:
    if tensor.startswith(b"\x93NUMPY"):
        bytes_io = BytesIO(tensor)
        ndarray_deserialized = np.load(bytes_io, allow_pickle=True)
        return cast(NDArray, ndarray_deserialized)
    return tensor


def ndarrays_to_parameters_custom(ndarrays: NDArrays) -> Parameters:
    tensors = [ndarray_to_bytes_custom(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_ndarrays_custom(parameters: Parameters) -> NDArrays:
    return [bytes_to_ndarray_custom(tensor) for tensor in parameters.tensors]


# Register the custom adapters
fl.common.parameter.ndarrays_to_parameters = ndarrays_to_parameters
fl.common.parameter.parameters_to_ndarrays = parameters_to_ndarrays
fl.common.parameter.ndarray_to_bytes = ndarray_to_bytes
fl.common.parameter.bytes_to_ndarray = bytes_to_ndarray

fl.common.ndarrays_to_parameters = ndarrays_to_parameters
fl.common.parameters_to_ndarrays = parameters_to_ndarrays

flwr.client.numpy_client.ndarrays_to_parameters = ndarrays_to_parameters
flwr.client.numpy_client.parameters_to_ndarrays = parameters_to_ndarrays