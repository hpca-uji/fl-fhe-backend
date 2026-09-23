from src.nn.layers.encrypted_layer import EncryptedLayer
from src.nn.layers.layer import Layer
from typing import List, Tuple
from flwr.common.typing import NDArrays
from functools import reduce
import numpy as np


def deserialized_layer(name_layer, weight_array, he_backend):
    if isinstance(weight_array, bytes):
        return EncryptedLayer(name_layer, he_backend.deserialize(weight_array), he_backend)

    elif hasattr(weight_array, 'decrypt') or hasattr(weight_array, 'serialize'):
        return EncryptedLayer(name_layer, weight_array, he_backend)

    else:
        return Layer(name_layer, weight_array)


def deserialized_model(client_query, he_backend):
    return [deserialized_layer(name_layer, weight_array, he_backend) for name_layer, weight_array in client_query.items()]


import time
from typing import List, Tuple, Dict, Union, Any

#Todo(DQ) Change this to a generic function and avoid NDArrays which is from Flower
# Or implement a mapper
def aggregate_custom(
    results: List[Tuple[NDArrays, int]], 
    he_backend=None,
    return_timing: bool = False
) -> Union[NDArrays, Tuple[NDArrays, Dict[str, float]]]:
    num_examples_total = sum([num_examples for _, num_examples in results])
    
    # Pre-calculate the fraction to reduce the homomorphic multiplication depth from 2 to 1.
    # This prevents FHE "scale out of bounds" errors and is computationally faster.
    t0_mul = time.perf_counter()
    weighted_weights = []
    for weights, num_examples in results:
        scalar = num_examples / num_examples_total
        if he_backend and hasattr(he_backend, 'enc_mul'):
            weighted_weights.append([he_backend.enc_mul(layer, scalar) for layer in weights])
        else:
            weighted_weights.append([layer * scalar for layer in weights])
    mul_time = time.perf_counter() - t0_mul
    
    t0_add = time.perf_counter()
    weights_prime: NDArrays = []
    for layer_updates in zip(*weighted_weights):
        if he_backend and hasattr(he_backend, 'enc_add'):
            weights_prime.append(reduce(he_backend.enc_add, layer_updates))
        else:
            weights_prime.append(reduce(np.add, layer_updates))
    add_time = time.perf_counter() - t0_add
            
    if return_timing:
        timing = {
            "homomorphic_mult_time": mul_time if (he_backend and hasattr(he_backend, 'enc_mul')) else 0.0,
            "homomorphic_add_time": add_time if (he_backend and hasattr(he_backend, 'enc_add')) else 0.0,
            "homomorphic_addition_time": (mul_time + add_time) if (he_backend and (hasattr(he_backend, 'enc_add') or hasattr(he_backend, 'enc_mul'))) else 0.0
        }
        return weights_prime, timing
        
    return weights_prime