import tenseal as ts
import numpy as np
import torch
import pickle
from io import BytesIO
from typing import cast
from flwr.common.typing import NDArrays, NDArray, Parameters
import flwr as fl

def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    return ndarrays_to_parameters_custom(ndarrays)


def parameters_to_ndarrays(parameters: Parameters, secret_path) -> NDArrays:
    with open(secret_path, 'rb') as f:
        query = pickle.load(f)

    context_client = ts.context_from(query["contexte"])
    return parameters_to_ndarrays_custom(parameters, context_client=context_client)


def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    return ndarray_to_bytes_custom(ndarray)


def bytes_to_ndarray(tensor: bytes, context_client) -> NDArray:
    bytes_io = BytesIO(tensor)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(NDArray, ndarray_deserialized)


def ndarray_to_bytes_custom(ndarray: NDArray) -> bytes:
    if type(ndarray) == ts.tensors.CKKSTensor:
        print(ndarray.serialize())
        return ndarray.serialize()

    bytes_io = BytesIO()
    np.save(bytes_io, 
            ndarray.cpu().detach().numpy() if isinstance(ndarray, torch.Tensor) else ndarray, 
            allow_pickle=False
            )
    return bytes_io.getvalue()


def bytes_to_ndarray_custom(tensor: bytes, context_client) -> NDArray:
    try:
        ndarray_deserialized = ts.ckks_tensor_from(context_client, tensor)
    except:
        bytes_io = BytesIO(tensor)
        ndarray_deserialized = np.load(bytes_io, allow_pickle=False)

    return cast(NDArray, ndarray_deserialized)


def ndarrays_to_parameters_custom(ndarrays: NDArrays) -> Parameters:
    tensors = [ndarray_to_bytes_custom(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_ndarrays_custom(parameters: Parameters, context_client) -> NDArrays:
    return [bytes_to_ndarray_custom(tensor, context_client) for tensor in parameters.tensors]


# Register the custom adapters
fl.common.parameter.ndarrays_to_parameters = ndarrays_to_parameters
fl.common.parameter.paramaters_to_ndarrays = parameters_to_ndarrays
fl.common.parameter.ndarray_to_bytes = ndarray_to_bytes
fl.common.parameter.bytes_to_ndarray = bytes_to_ndarray