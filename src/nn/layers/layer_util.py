from src.nn.layers.encrypted_layer import EncryptedLayer
from src.nn.layers.layer import Layer
from typing import List, Tuple
from flwr.common.typing import NDArrays
from functools import reduce
import tenseal as ts
import numpy as np


def deserialized_layer(name_layer, weight_array, context):
    if isinstance(weight_array, bytes):
        return EncryptedLayer(name_layer, ts.ckks_tensor_from(context, weight_array), context)

    elif type(weight_array) == ts.tensors.CKKSTensor:
        return EncryptedLayer(name_layer, weight_array, context)

    else:
        return Layer(name_layer, weight_array)


def deserialized_model(client_query, context):
    return [deserialized_layer(name_layer, weight_array, context) for name_layer, weight_array in client_query.items()]


#Todo(DQ) Change this to a generic function and avoid NDArrays which is from Flower
# Or implement a mapper
def aggregate_custom(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    num_examples_total = sum([num_examples for _, num_examples in results])
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) * (1/num_examples_total)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime