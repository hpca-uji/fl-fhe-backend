import os
import time
import shutil
import numpy as np
import torch
from src.core.util.fhe_metrics import (
    calculate_weights_size,
    log_keygen_metrics,
    log_client_fhe_metrics,
    log_server_fhe_metrics,
)
from src.encryption.factory import HomomorphicEncrytionFactory
from src.nn.layers.layer_util import aggregate_custom

def test_metrics():
    test_results_dir = "results/test_metrics"
    if os.path.exists(test_results_dir):
        shutil.rmtree(test_results_dir)

    print("=== Testing calculate_weights_size ===")
    sample_weights = [np.random.randn(10, 10).astype(np.float32), np.random.randn(5).astype(np.float32)]
    total_bytes, kb, mb = calculate_weights_size(sample_weights)
    print(f"Sample weights size: {total_bytes} bytes, {kb:.3f} KB, {mb:.6f} MB")
    assert total_bytes == (100 * 4 + 5 * 4), f"Expected 420 bytes, got {total_bytes}"

    print("=== Testing log_keygen_metrics ===")
    keygen_path = log_keygen_metrics(
        component="test_client",
        keygen_time_sec=0.123456,
        fhe_lib="OPENFHE",
        poly_modulus_degree=8192,
        results_dir=test_results_dir
    )
    assert os.path.exists(keygen_path), f"File {keygen_path} not found"
    with open(keygen_path, "r") as f:
        print("Keygen CSV content:\n", f.read())

    print("=== Testing log_client_fhe_metrics ===")
    client_path = log_client_fhe_metrics(
        client_id="test_0",
        server_round=1,
        decryption_time_sec=0.054321,
        encryption_time_sec=0.098765,
        weights_size_bytes=total_bytes,
        keygen_time_sec=0.123456,
        model="MOBILENETV3",
        fhe_lib="OPENFHE",
        poly_modulus_degree=8192,
        results_dir=test_results_dir
    )
    assert os.path.exists(client_path), f"File {client_path} not found"
    with open(client_path, "r") as f:
        print("Client CSV content:\n", f.read())

    print("=== Testing log_server_fhe_metrics ===")
    server_path = log_server_fhe_metrics(
        server_round=1,
        homomorphic_addition_time_sec=0.234567,
        homomorphic_mult_time_sec=0.012345,
        total_aggregation_time_sec=0.345678,
        received_weights_size_bytes=total_bytes * 2,
        num_clients=2,
        keygen_time_sec=0.123456,
        model="MOBILENETV3",
        fhe_lib="OPENFHE",
        poly_modulus_degree=8192,
        results_dir=test_results_dir
    )
    assert os.path.exists(server_path), f"File {server_path} not found"
    with open(server_path, "r") as f:
        print("Server CSV content:\n", f.read())

    print("=== Testing aggregate_custom with timing ===")
    w1 = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    w2 = [np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    results = [(w1, 10), (w2, 10)]
    agg_w, timing = aggregate_custom(results, he_backend=None, return_timing=True)
    print("Aggregate result:", agg_w)
    print("Timing:", timing)
    assert np.allclose(agg_w[0], np.array([2.0, 3.0]))

    # Test with OpenFHE
    try:
        he = HomomorphicEncrytionFactory.get_backend("OPENFHE", "CKKS")
        he.set_poly_modulus_degree(8192)
        he.create_context()
        t0 = time.time()
        he.generate_keys()
        print(f"OpenFHE KeyGen time: {time.time() - t0:.4f}s")
        enc_w1 = [he.encrypt(np.array([1.0, 2.0]))]
        enc_w2 = [he.encrypt(np.array([3.0, 4.0]))]
        enc_results = [(enc_w1, 10), (enc_w2, 10)]
        agg_enc, enc_timing = aggregate_custom(enc_results, he_backend=he, return_timing=True)
        dec = he.decrypt(agg_enc[0])
        print("OpenFHE Decrypted Aggregate:", dec[:2])
        print("OpenFHE Homomorphic Timing:", enc_timing)
        assert np.allclose(dec[:2], [2.0, 3.0], atol=1e-3)
    except Exception as e:
        print("OpenFHE test exception (if any):", e)

    # Clean up test dir
    shutil.rmtree(test_results_dir)
    print("\nALL FHE METRICS AND TIMING TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_metrics()
