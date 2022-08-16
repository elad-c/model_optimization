# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
if tf.__version__ < "2.6":
    from tensorflow.python.keras.layers.core import TFOpLambda
    from tensorflow.keras.layers import Dense, Conv2D, Softmax, Concatenate, Reshape, Permute
else:
    from keras.layers.core import TFOpLambda
    from keras.layers import Dense, Conv2D, Softmax, Concatenate, Reshape, Permute

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph, BaseNode, OutTensor
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.common.constants import REUSE, REUSE_GROUP
from model_compression_toolkit.core.keras.reader.node_builder import REUSED_IDENTIFIER
from model_compression_toolkit.core.keras.constants import KERNEL, BIAS, USE_BIAS, NUM_HEADS, KEY_DIM, VALUE_DIM, \
    QUERY_SHAPE, KEY_SHAPE, VALUE_SHAPE, OUTPUT_SHAPE, ATTENTION_AXES, ACTIVATION, LINEAR, FILTERS, \
    FUNCTION, DIMS, TARGET_SHAPE, F_STRIDED_SLICE, F_STACK, Q_KERNEL, Q_BIAS, K_KERNEL, K_BIAS, V_KERNEL, V_BIAS, \
    OUTPUT_KERNEL, OUTPUT_BIAS, F_MATMUL, TRANSPOSE_B, KERNEL_SIZE, AXIS, F_STRIDED_SLICE_BEGIN, F_STRIDED_SLICE_END


class DenseToConv(common.BaseSubstitution):
    """
    Removes a MultiHeadAttention node from the graph,
    and replaces it with a compatible graph that consists of Conv2D,
    tf.matmul, Softmax and Concatenate layers
    """

    def __init__(self):
        """
        Matches MultiHeadAttention node.
        """
        super().__init__(matcher_instance=NodeOperationMatcher(Dense))

    @staticmethod
    def _get_weight_by_name(mha_node, w_str):
        return [k for k in mha_node.weights.keys() if w_str in k][0]

    def substitute(self,
                   graph: Graph,
                   dense_node: BaseNode) -> Graph:
        """
        TODO: edit doc + add tests
        Removes a MultiHeadAttention node from the graph, and replaces it with
         a compatible graph that consists of Dense, strided_slice, Dot, Softmax and Concatenate layers.
        Additional reshape and permute nodes are used to shape the inputs to the standard
        of [B, Iters, Sequence, C]. All attention axes are folded on the Sequence axis and iteration axes
        to the Iters axis.

        Args:
            graph: Graph we apply the substitution on.
            mha_node: MultiHeadAttention node to replace.

        Returns:
            Graph after applying the substitution.
        """

        if len(dense_node.input_shape) == 2:
            return graph
        elif len(dense_node.input_shape) != 4:
            raise NotImplemented

        k = dense_node.weights[self._get_weight_by_name(dense_node, KERNEL)].copy()
        b = dense_node.weights[self._get_weight_by_name(dense_node, BIAS)].copy()
        k = k[np.newaxis, np.newaxis, ...]

        _reuse_params = {REUSE: dense_node.reuse, REUSE_GROUP: dense_node.reuse_group}
        conv_node = BaseNode(dense_node.name, {FILTERS: b.shape[0], KERNEL_SIZE: 1,
                                               USE_BIAS: dense_node.framework_attr[USE_BIAS],
                                               ACTIVATION: dense_node.framework_attr[ACTIVATION]},
                             dense_node.input_shape, dense_node.output_shape, {KERNEL: k, BIAS: b}, Conv2D,
                             **_reuse_params)
        graph.add_node(conv_node)

        # replace Dense node with Conv node
        _in_edge = list(graph.in_edges(dense_node))[0]
        _out_edges = graph.out_edges(dense_node)
        graph.add_edge(_in_edge[0], conv_node, **graph.get_edge_data(*_in_edge, 0))
        graph.remove_edge(_in_edge[0], dense_node)
        graph.reconnect_out_edges(current_node=dense_node, new_node=conv_node)

        # Finally, remove the Dense node
        graph.remove_node(dense_node, new_graph_outputs=[OutTensor(conv_node, 0)])

        return graph
