#!/usr/bin/env python3
# Copyright 2018 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
@title           :layers/tf_layers.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/10/2018
@version         :1.0
@python_version  :3.6.6

A collection of wrappers for tensorflow layers.
"""

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from misc.custom_exceptions.argument_exception import ArgumentException

def fully_connected(inputs, num_outputs, activation_fn=tf.nn.relu, \
                    weights_initializer=initializers.xavier_initializer(), \
                    biases_initializer=tf.zeros_initializer(), name=None, \
                    use_bn=False, drop_rate=None, is_training=None):
    """A fully-connected layer.

    Creates a fully connected layer, that may use batch norm and/or dropout.

    Args:
        inputs: See docs of "tf.contrib.layers.fully_connected".
        num_outputs: See docs of "tf.contrib.layers.fully_connected".
        activation_fn (optional): See docs of
            "tf.contrib.layers.fully_connected".
        weights_initializer (optional): See docs of
            "tf.contrib.layers.fully_connected".
        biases_initializer (optional): See docs of
            "tf.contrib.layers.fully_connected".
        name (optional): The layer "tf.contrib.layers.fully_connected" has no
            name attribute. Thus, the name will not affect the fully-connected
            layer. However, if used, the added batch_norm and dropout layer
            will have this string as a prefix in their names.
        use_bn (default: False): Whether batch normalization will be used in
            this layer.
            Note, that you need to explicitly add the moving average ops
            to the update ops of your graph!
        drop_rate (optional): If not None, this option enables dropout. See
            docs of  "tf.layers.dropout" (parameter "rate") for a proper
            description of the parameter.
        is_training: This option has to be provided, if either of the
            parameters "use_bn" or "drop_rate" is provided.
            This option should be True (or evaluate to True), if the network
            is in training mode. Note, that batch norm and dropout behave
            differently in training and inference mode.

    Returns:
        A tensor (output tensor of the layer).
    """
    layer = tf.contrib.layers.fully_connected(inputs, num_outputs, 
        activation_fn=activation_fn, weights_initializer=weights_initializer,
        biases_initializer=biases_initializer)

    return _add_bn_and_dropout(layer, name=name, use_bn=use_bn,
                               drop_rate=drop_rate, is_training=is_training)

def conv2d_transpose(inputs, filters, kernel_size, strides=(1, 1),
                     padding='valid', activation=None, use_bias=True,
                     kernel_initializer=None, name=None, use_bn=False,
                     drop_rate=None, is_training=None):
    """A transpose convolutional layer.

    Creates a deconvolutional layer, that may use batch norm and/or dropout.

    Args:
        inputs: See docs of "tf.layers.conv2d_transpose".
        filters: See docs of "tf.layers.conv2d_transpose".
        kernel_size: See docs of "tf.layers.conv2d_transpose".
        padding (optional): See docs of "tf.layers.conv2d_transpose".
        activation (optional): See docs of "tf.layers.conv2d_transpose".
        use_bias (default: True): See docs of "tf.layers.conv2d_transpose".
        kernel_initializer (optional): See docs of "tf.layers.conv2d_transpose".
        name (optional): See docs of "tf.layers.conv2d_transpose".
            See also the comments of this parameter in the docstring of the
            method 'fully_connected'.
        use_bn: See docs of method "fully_connected".
        drop_rate: See docs of method "fully_connected".
        is_training: See docs of method "fully_connected".

    Returns:
        A tensor (output tensor of the layer).
    """
    layer = tf.layers.conv2d_transpose(inputs, filters, kernel_size,
                                       strides=strides, padding=padding,
                                       activation=activation,
                                       kernel_initializer=kernel_initializer,
                                       name=name, use_bias=use_bias)

    return _add_bn_and_dropout(layer, name=name, use_bn=use_bn,
                               drop_rate=drop_rate, is_training=is_training)

def conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='valid',
           activation=None, use_bias=True, kernel_initializer=None, name=None,
           pool_size=None, pool_strides=None, pool_padding='valid',
           pool_average=False, use_bn=False, drop_rate=None, is_training=None):
    """A convolutional layer.

    Creates a convolutional layer, that may use max- or average-pooling and may
    use batch norm and/or dropout.

    Args:
        inputs: See docs of "tf.layers.conv2d".
        filters: See docs of "tf.layers.conv2d".
        kernel_size: See docs of "tf.layers.conv2d".
        padding (optional): See docs of "tf.layers.conv2d".
        activation (optional): See docs of "tf.layers.conv2d".
        use_bias (default: True): See docs of "tf.layers.conv2d".
        kernel_initializer (optional): See docs of "tf.layers.conv2d".
        name (optional): See docs of "tf.layers.conv2d".
            See also the comments of this parameter in the docstring of the
            method 'fully_connected'.
            If pooling is used, the given string will be a prefix of this
            operation.
        pool_size: If given, a pooling layer will be added. See docs of
            "tf.layers.max_pooling2d".
        pool_strides: If given, a pooling layer will be added. See docs of
            "tf.layers.max_pooling2d" (parameter "strides").
        pool_padding (default: 'valid'): See docs of "tf.layers.max_pooling2d"
            (parameter "padding").
        pool_average (default: False): If pooling is activated, this option
            decides wether to use max or average pooling.
        use_bn: See docs of method "fully_connected".
        drop_rate: See docs of method "fully_connected".
        is_training: See docs of method "fully_connected".

    Returns:
        A tensor (output tensor of the layer).
    """
    layer = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides,
                             padding=padding, activation=activation,
                             kernel_initializer=kernel_initializer, name=name,
                             use_bias=use_bias)

    if pool_size is not None or pool_strides is not None:
        assert(pool_size is not None and pool_strides is not None)

        pool_name = None
        if name is not None:
            pool_name = name + '_pool'

        if pool_average:
            layer = tf.layers.average_pooling2d(layer, pool_size, pool_strides,
                                                padding=pool_padding,
                                                name=pool_name)
        else:
            layer = tf.layers.max_pooling2d(layer, pool_size, pool_strides,
                                            padding=pool_padding,
                                            name=pool_name)

    return _add_bn_and_dropout(layer, name=name, use_bn=use_bn,
                               drop_rate=drop_rate, is_training=is_training)

def _add_bn_and_dropout(layer, name, use_bn, drop_rate,
                        is_training):
    """A helper method, to add batch-norm and or dropout to a given tensor.

    Args:
        layer: The input tensor.

        For the remaining arguments, please see the documentation of the method
        "fully_connected".

    Returns:
        A tensor, that may have added a batchnorm and/or dropout layer to the
        input tensor.

    """
    name_bn = None
    name_do = None
    if name is not None:
        name_bn = name + '_bn'
        name_do = name + '_dropout'

    if use_bn or drop_rate is not None:
        if is_training is None:
            raise ArgumentException('Please specify, whether the network is '
                                    + 'in training or inference mode.')
    if is_training is not None:
        assert(use_bn or drop_rate is not None)

    if use_bn:
        layer = tf.layers.batch_normalization(layer, center=True, scale=True,
                                              training=is_training,
                                              name=name_bn)

    if drop_rate is not None:
        layer = tf.layers.dropout(layer, rate=drop_rate, training=is_training,
                                  name=name_do)

    return layer

if __name__ == '__main__':
    pass


