#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import numpy as np

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit
import pycuda.driver as cuda

import sys
import tensorrt as trt

# from model import TRT_MODEL_PATH
WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))
MODEL_DIR  = os.path.join(WORKING_DIR, "models")
TRT_MODEL_PATH = os.path.join(MODEL_DIR, "average_layer.onnx")

from load_plugin_lib import load_plugin_lib

# ../common.py
parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(1, parent_dir)
import common

# Reuse some BiDAF-specific methods
# ../engine_refit_onnx_bidaf/data_processing.py
# sys.path.insert(1, os.path.join(parent_dir, 'engine_refit_onnx_bidaf'))
# from engine_refit_onnx_bidaf.data_processing import preprocess, get_inputs

# Maxmimum number of words in context or query text.
# Used in optimization profile when building engine.
# Adjustable.
MAX_TEXT_LENGTH = 64

# WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

# Path to which trained model will be saved (check README.md)
ENGINE_FILE_PATH = os.path.join(MODEL_DIR, 'average_layer.trt')

# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Builds TensorRT Engine
def build_engine(model_path):

# Load pretrained model
    if not os.path.isfile(TRT_MODEL_PATH):
        raise IOError(
            "\n{}\n{}\n{}\n".format(
                "Failed to load model file ({}).".format(TRT_MODEL_PATH),
                "Please use 'python3 model.py' to generate the ONNX model.",
                "For more information, see README.md",
            )
        )

    if os.path.exists(ENGINE_FILE_PATH):
        print(f"Loading saved TRT engine from {ENGINE_FILE_PATH}")
        with open(ENGINE_FILE_PATH, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            runtime.max_threads = 10
            engine = runtime.deserialize_cuda_engine(f.read())
            return engine
    else:
        print("Engine plan not saved. Building new engine...")
        
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    runtime = trt.Runtime(TRT_LOGGER)

    # Parse model file
    print("Loading ONNX file from path {}...".format(model_path))
    with open(model_path, "rb") as model:
        print("Beginning ONNX file parsing")
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing of ONNX file")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))

    # The input text length is variable, so we need to specify an optimization profile.
    profile = builder.create_optimization_profile()
    # for i in range(network.num_inputs):
    #     input = network.get_input(i)
    #     assert input.shape[0] == -1
    #     min_shape = [1] + list(input.shape[1:])
    #     opt_shape = [8] + list(input.shape[1:])
    #     max_shape = [MAX_TEXT_LENGTH] + list(input.shape[1:])
    #     profile.set_shape(input.name, min_shape, opt_shape, max_shape)
    # config.add_optimization_profile(profile)

    print("Building TensorRT engine. This may take a few minutes.")
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)
    with open(ENGINE_FILE_PATH, "wb") as f:
        f.write(plan)
    return engine

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            # Our input tensors have dynamic shape.
            # We only added 1 optimization profile, its index is 0
            profile = 0
            # -1th profile contains the MAX size tensor shape
            max_shape = engine.get_tensor_profile_shape(binding, profile)[-1]
        else:
            # The output tensor sizes are not dynamic
            max_shape = engine.get_tensor_shape(binding)
        size = trt.volume(max_shape)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(common.HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(common.HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def custom_plugin_impl(input_arr_0, input_arr_1, engine):
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()
    inputs[0].host = input_arr_0.astype(trt.nptype(trt.float32)).ravel()
    inputs[1].host = input_arr_1.astype(trt.nptype(trt.float32)).ravel()
    trt_outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )
    output = trt_outputs[0]
    return output

def main():
    # Load the shared object file containing the Hardmax plugin implementation.
    # By doing this, you will also register the Hardmax plugin with the TensorRT
    # PluginRegistry through use of the macro REGISTER_TENSORRT_PLUGIN present
    # in the plugin implementation. Refer to plugin/customHardmaxPlugin.cpp for more details.
    load_plugin_lib()
    
    engine = build_engine(TRT_MODEL_PATH)

    axis = 0
    shape = [1,4]

    arr_0 = np.random.rand(*shape)
    arr_0 = (arr_0 - 0.5) * 200
    
    arr_1 = np.random.rand(*shape)
    arr_1 = (arr_1 - 0.5) * 200

    res = custom_plugin_impl(arr_0, arr_1, engine)
    print(f'arr_0 = {arr_0}')
    print(f'arr_1 = {arr_1}')
    print(f'res = {res}')

    print("Passed")

if __name__ == "__main__":
    main()
