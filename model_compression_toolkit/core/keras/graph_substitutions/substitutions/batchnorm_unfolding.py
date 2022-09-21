# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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


import copy
import numpy as np
from typing import Tuple, Callable

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.graph.base_graph import Graph

import numpy as np
from tensorflow.keras.layers import BatchNormalization, DepthwiseConv2D, Conv2DTranspose, Conv2D, Dense

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher
from model_compression_toolkit.core.keras.constants import KERNEL, LINEAR, ACTIVATION, DEPTHWISE_KERNEL, BIAS, GAMMA, BETA, \
    MOVING_MEAN, MOVING_VARIANCE, EPSILON, USE_BIAS, LAYER_NAME


class BatchNormalizationUnFolding(common.BaseSubstitution):
    """
    UnFold BatchNormalization into preceding linear layers.
    """

    def __init__(self):
        """
        Matches: A linear Operation that might contain a folded Batch Normalization node.
        """

        super().__init__(matcher_instance=NodeOperationMatcher(DepthwiseConv2D) |
                                          NodeOperationMatcher(Conv2D) |
                                          NodeOperationMatcher(Conv2DTranspose) |
                                          NodeOperationMatcher(Dense))

    def substitute_(self,
                   graph: Graph,
                   linear_op_node: BaseNode) -> Graph:
        """
        Unfold BatchNormalization after a linear layer if prior info exists.

        Args:
            graph: Graph we apply the substitution on.
            linear_op_node: Linear op node.

        Returns:
            Graph after applying the substitution.
        """

        std_out = linear_op_node.prior_info.std_output
        mean_out = linear_op_node.prior_info.mean_output
        if std_out is None and mean_out is None:
            # A linear node with no prior info => skip substitution
            return graph

        eps = 0.0001

        # std_out[:] = 1.0
        # mean_out[:] = 0.0
        # eps = 0.1

        var_out = np.square(std_out)
        bn_weights = {
            GAMMA: np.sqrt(var_out + eps),
            BETA: mean_out,
            MOVING_MEAN: mean_out,
            MOVING_VARIANCE: var_out,
        }

        bn_node = BaseNode(linear_op_node.name + '_qat_bn', {EPSILON: eps, 'momentum': 0.99},
                           linear_op_node.output_shape, linear_op_node.output_shape, bn_weights,
                           BatchNormalization, reuse=linear_op_node.reuse, reuse_group=linear_op_node.reuse_group
                           )

        graph.add_node(bn_node)
        bn_node.final_activation_quantization_cfg = linear_op_node.final_activation_quantization_cfg
        linear_op_node.final_activation_quantization_cfg.enable_activation_quantization = False
        graph.reconnect_out_edges(current_node=linear_op_node, new_node=bn_node)
        graph.add_edge(linear_op_node, bn_node, source_index=0, sink_index=0)

        return graph

    def substitute(self,
                   graph: Graph,
                   linear_op_node: BaseNode) -> Graph:
        """
        Unfold BatchNormalization after a linear layer if prior info exists.

        Args:
            graph: Graph we apply the substitution on.
            linear_op_node: Linear op node.

        Returns:
            Graph after applying the substitution.
        """

        std_out = linear_op_node.prior_info.std_output
        mean_out = linear_op_node.prior_info.mean_output
        if std_out is None and mean_out is None:
            # A linear node with no prior info => skip substitution
            return graph

        eps = 0.0001

        # std_out[:] = 1.0
        # mean_out[:] = 0.0
        # eps = 0.1

        var_out = np.square(std_out)
        bn_weights = {
            MOVING_MEAN: mean_out,
            MOVING_VARIANCE: var_out,
        }

        bn_node = BaseNode(linear_op_node.name + '_qat_bn', {EPSILON: eps, 'momentum': 0.99, 'scale': False, 'center': False},
                           linear_op_node.output_shape, linear_op_node.output_shape, bn_weights,
                           BatchNormalization, reuse=linear_op_node.reuse, reuse_group=linear_op_node.reuse_group
                           )

        graph.add_node(bn_node)

        bn_weights = {
            GAMMA: np.sqrt(var_out + eps),
            BETA: mean_out,
            MOVING_MEAN: np.zeros(mean_out.shape),
            MOVING_VARIANCE: np.ones(var_out.shape),
        }
        bn_node2 = BaseNode(linear_op_node.name + '_qat_gammabeta', {EPSILON: 0.0},
                           linear_op_node.output_shape, linear_op_node.output_shape, bn_weights,
                           BatchNormalization, reuse=linear_op_node.reuse, reuse_group=linear_op_node.reuse_group
                           )
        graph.add_node(bn_node2)

        bn_node.final_activation_quantization_cfg = linear_op_node.final_activation_quantization_cfg
        bn_node2.final_activation_quantization_cfg = linear_op_node.final_activation_quantization_cfg
        linear_op_node.final_activation_quantization_cfg.enable_activation_quantization = False
        bn_node.final_activation_quantization_cfg.enable_activation_quantization = False
        graph.reconnect_out_edges(current_node=linear_op_node, new_node=bn_node2)
        graph.add_edge(linear_op_node, bn_node, source_index=0, sink_index=0)
        graph.add_edge(bn_node, bn_node2, source_index=0, sink_index=0)

        return graph
