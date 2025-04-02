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
import onnx
import onnx_graphsurgeon as gs

def main():
    input0 = gs.Variable(name="input0", dtype=np.float32, shape=(1, 4))
    input1 = gs.Variable(name="input1", dtype=np.float32, shape=(1, 4))
    output = gs.Variable(name="output", dtype=np.float32, shape=(1, 4))

    node = gs.Node(op="AveragePlugin", inputs=[input0, input1], outputs=[output], attrs={"axis": 0})
    # node = gs.Node(op="Add", inputs=[input0, input1], outputs=[output], attrs={"axis": 0})

    graph = gs.Graph(nodes=[node], inputs=[input0, input1], outputs=[output])

    model = gs.export_onnx(graph)
    onnx.save(model, "average_layer.onnx")    

if __name__ == '__main__':
    main()