#!/usr/bin/env python

import argparse
import sys

import numpy as np
import tritonclient.grpc as grpcclient

if __name__ == "__main__":
    try:
        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001",
            verbose=False,
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "global_dnn"

    # Infer
    inputs = []
    outputs = []

    # Data for each input
    input_data = {
        "A": np.array([[1.0]], dtype=np.float32),
        "B": np.array([["string1"]], dtype=object),
        "C": np.array([[2.0]], dtype=np.float32),
        "D": np.array([[3.0]], dtype=np.float32),
        "E": np.array([[4.0]], dtype=np.float32),
        "F": np.array([["string2"]], dtype=object),
        "G": np.array([[5.0]], dtype=np.float32),
        "H": np.array([[6.0]], dtype=np.float32),
        "I": np.array([["string3"]], dtype=object),
        "J": np.array([[7.0]], dtype=np.float32),
        "K": np.array([[8.0]], dtype=np.float32),
        "L": np.array([["string4"]], dtype=object),
        "M": np.array([[9.0]], dtype=np.float32),
        "N": np.array([[10.0]], dtype=np.float32),
        "O": np.array([["string5"]], dtype=object),
        "P": np.array([[11.0]], dtype=np.float32),
        "Q": np.array([[12.0]], dtype=np.float32),
        "R": np.array([["string6"]], dtype=object),
        "S": np.array([["string7"]], dtype=object),
        "T": np.array([[13.0]], dtype=np.float32),
        "U": np.array([[14.0]], dtype=np.float32),
        "V": np.array([[15.0]], dtype=np.float32),
        "W": np.array([[16.0]], dtype=np.float32)
    }
    
    # Add the input data to the Triton client input list
    for name, value in input_data.items():
        input_tensor = grpcclient.InferInput(name, value.shape, "FP32" if value.dtype == np.float32 else "BYTES")
        input_tensor.set_data_from_numpy(value)
        inputs.append(input_tensor)

    # Define the output
    outputs.append(grpcclient.InferRequestedOutput("output"))

    if not triton_client.is_model_ready(model_name):
        triton_client.load_model(model_name)

    # Test with outputs
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        client_timeout=0.005, ## low
        headers={"test": "1"},
    )

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    print(statistics)

    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
    
    if triton_client.is_model_ready(model_name):
        triton_client.unload_model(model_name)

    # Get the output arrays from the results
    output0_data = results.as_numpy("output")
    print("Output:", output0_data)
    print("PASS: infer")
