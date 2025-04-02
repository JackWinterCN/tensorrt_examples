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

import numpy as np
import os
import sys
import tensorrt as trt

# ../common.py
parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(1, parent_dir)
import common

from load_plugin_lib import load_plugin_lib

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

def reference_impl(input_arr_0, input_arr_1, axis = 0):
   return (input_arr_0 + input_arr_1) / 2.0

def make_trt_network_and_engine(input_shape, axis):
    registry = trt.get_plugin_registry()
    plugin_creator = registry.get_plugin_creator("AveragePlugin", "1")
    axis_buffer = np.array([axis])
    axis_attr = trt.PluginField("axis", axis_buffer, type=trt.PluginFieldType.INT32)
    field_collection = trt.PluginFieldCollection([axis_attr])
    plugin = plugin_creator.create_plugin(name="AveragePlugin", field_collection=field_collection)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)

    input_layer_0 = network.add_input(name="input_layer_0", dtype=trt.float32, shape=input_shape)
    input_layer_1 = network.add_input(name="input_layer_1", dtype=trt.float32, shape=input_shape)
    plugin_layer = network.add_plugin_v2(inputs=[input_layer_0, input_layer_1], plugin=plugin)
    network.mark_output(plugin_layer.get_output(0))

    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)

    return engine

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
    load_plugin_lib()
    axis = 0
    shape = [1,4]

    arr_0 = np.random.rand(*shape)
    arr_0 = (arr_0 - 0.5) * 200
    
    arr_1 = np.random.rand(*shape)
    arr_1 = (arr_1 - 0.5) * 200
    
    engine = make_trt_network_and_engine(shape, axis)
    res2 = custom_plugin_impl(arr_0, arr_1, engine)
    ref_res = reference_impl(arr_0, arr_1)
    print(f'arr_0 = {arr_0}')
    print(f'arr_1 = {arr_1}')
    print(f'res2 = {res2}')
    print(f'ref_res = {ref_res}')
    

if __name__ == '__main__':
    main()
