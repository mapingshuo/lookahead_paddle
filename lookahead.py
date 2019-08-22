# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
from collections import defaultdict

from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.framework import Program, Variable, name_scope, default_main_program, default_startup_program

from paddle.fluid import framework
from paddle.fluid import layers
from paddle.fluid import unique_name
from paddle.fluid.backward import append_backward, _some_in_set_, _append_grad_suffix_
from paddle.fluid.clip import append_gradient_clip_ops, error_clip_callback
from paddle.fluid.framework import program_guard
from paddle.fluid.initializer import Constant
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import ops
from paddle.fluid.regularizer import append_regularization_ops
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay
from paddle.fluid import core
from paddle.fluid.layers import tensor
from functools import reduce
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager
from paddle import compat as cpt

__all__ = [
    'LookaheadOptimizer'
]

class LookaheadOptimizer(object):
    def __init__(self,
                 inner_optimizer,
                 alpha=0.5,
                 k=5,
                 ignore_embed=False,
                 name=None):
        assert inner_optimizer is not None
        self.inner_optimizer = inner_optimizer
        self.alpha = alpha
        self.k = k
        self.ignore_embed = ignore_embed
        self.type = "lookahead"

    def minimize(self, loss, startup_program=None):

        # Apply SGD to the main_program
        mini_out = self.inner_optimizer.minimize(loss)

        # Get startup_program and main_program
        if startup_program is None:
            startup_program = default_startup_program()
        main_block = loss.block
        main_program = main_block.program

        # add some vars to the main_program
        params = [param.name for param in main_block.all_parameters()]
        if self.ignore_embed:
            params = [x for x in params if "embedding" not in x]
        # ignore batch norm
        # params = [x for x in params if "batch_norm" not in x]
        for param in params:
            main_block.create_var(
                name=param + "@SLOW",
                shape=main_block.var(param).shape,
                dtype=main_block.var(param).dtype,
                persistable=True)

        # add some vars to the startup_program
        startup_block = startup_program.global_block()
        for param in params:
            assert (startup_block.var(param) is not None)
            startup_block.create_var(
                name=param + "@SLOW",
                shape=startup_block.var(param).shape,
                dtype=startup_block.var(param).dtype,
                persistable=True)

            startup_block.append_op(
                type="assign",
                inputs={"X": startup_block.var(param)},
                outputs={"Out": startup_block.var(param + "@SLOW")})

        # Add Var k to main prog and startup prog
        k = layers.create_global_var(
            name="lookahead_k",
            shape=[1],
            value=int(self.k),
            dtype='int32',
            persistable=True)

        # Add Var alpha to main prog and startup prog
        alpha = layers.create_global_var(
            name="lookahead_alpha",
            shape=[1],
            value=float(self.alpha),
            dtype='float32',
            persistable=True)

        # Add Var step
        step = layers.create_global_var(
            name="lookahead_step",
            shape=[1],
            value=int(0),
            dtype='int32',
            persistable=True)
        layers.increment(x=step, value=1.0, in_place=True)

        # lookahead
        zero_var = layers.fill_constant(shape=[1], dtype='float32', value=0.0)

        one_var = layers.fill_constant(shape=[1], dtype='float32', value=1.0)

        block = main_block

        mod = layers.elementwise_mod(step, k)
        with layers.control_flow.Switch() as switch:
            with switch.case(mod == zero_var):
                for param_name in params:
                    tmp_var = layers.elementwise_add(
                        layers.elementwise_mul(block.var(param_name), alpha),
                        layers.elementwise_mul(
                            block.var(param_name + "@SLOW"),
                            layers.elementwise_sub(one_var, alpha)))
                    layers.assign(
                        input=tmp_var, output=block.var(param_name + "@SLOW"))
                    layers.assign(input=tmp_var, output=block.var(param_name))
            with switch.default():
                pass
        return mini_out
