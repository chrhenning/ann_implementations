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
@title           :networks/classifier/cnn_classifier.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/22/2018
@version         :1.0
@python_version  :3.6.6

An implementation of a classifier, that is composed of arbitrary many
convolutional layers (whereas each layer consists of an actual convolutional
layer plus a pooling layer for downsampling) and a final fully-connected layer.
The output of the network is evaluated via a cross-entropy through a softmax.
"""

import numpy as np
import tensorflow as tf
import os

from networks.network_base_tf import NetworkBaseTF
import misc.shared_vars as shared
from misc.custom_exceptions.custom_exception import CustomException
import layers.tf_layers as layers

import logging
logger = logging.getLogger(shared.logging_name)

class CNNClassifier(NetworkBaseTF):
    """An implementation of a multilayer CNN with max-pooling, and a final
    fully connected layer, that passes its outputs through a softmax into a
    cross-entropy loss.

    Attributes (additional to base class):
        num_layers: The number of layers in the network (a pair of
            convolutional + pooling layer are considered to be 1 layer).
            Example: A network with 4 layers will consist of 3 conv layers
            (each followed by a pooling layer) + 1 fully-connected layer.
        use_batchnorm (default: True): Whether the network should use batch
            normalization.
    """
    _DEFAULT_NUM_LAYERS = 4

    def __init__(self, mode='train'):
        """
        Args:
            mode: The value of the attribute "mode".
        """
        super().__init__(mode)

        # This network is not recurrent.
        if shared.data.sequence:
            raise CustomException("The network doesn't support sequence data.")

        # This is a classifier!
        if not shared.data.classification:
            raise CustomException("The network does only support " + \
                                  "classification datasets.")

        # We use 2D convolutions. So we need to ensure, that the input data
        # can be considered 2-dimensional.
        if not np.size(shared.data.in_shape) in [2, 3]:
            raise CustomException("The network does only support " + \
                                  "datasets with an input shape that is " + \
                                  "either 2D or 3D (with the 3rd " + \
                                  "dimension representing channels).")

        self._num_layers = CNNClassifier._DEFAULT_NUM_LAYERS

        self._compute_network_layout()

        self._use_batchnorm = True

    @property
    def num_layers(self):
        """Getter for the attribute num_layers."""
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        """Setter for the attribute num_layers.

        Note, that this setter will overwrite all layer-specific settings, that
        might have been already set.

        Here is how the default network is set up.
        Both, convolutional and pooling layers use "same" padding. While
        pooling layers have a stride of 2, convolutional layers have a stride
        of 2.

        The default kernel sizes are: 5x5 (conv layers) and 2x2 (pool layers).
        The number of filters is set to 32 in the first layer and is then
        doubled every layer.
        """
        if self._is_build:
            raise CustomException('The num_layers attribute can only be '
                                  + 'changed if the network has not been '
                                  + 'build yet.')
        self._num_layers = value

        self._compute_network_layout()

    @property
    def use_batchnorm(self):
        """Getter for the attribute use_batchnorm."""
        return self._use_batchnorm

    @use_batchnorm.setter
    def use_batchnorm(self, value):
        """Setter for the attribute use_batchnorm."""
        if self._is_build:
            raise CustomException('The use_batchnorm attribute can only be '
                                  + 'changed if the network has not been '
                                  + 'build yet.')
        self._use_batchnorm = value

    def get_num_filters(self, layer_ind):
        """Get the number output filter maps of a convolutional layer.

        Args:
            layer_ind: The index of the layer (between 0 and num_layers-2).
                Note, that the last layer is fully-connected.

        Returns:
            The number of filters of the requested layer.
        """
        assert(layer_ind < self.num_gen_layers-1)
        return self._gen_filters[layer_ind]

    def set_num_filters(self, num_filters, layer_ind):
        """Set the number output filter maps of a convolutional layer.

        Note, this method can only be called for a network, that has not been
        build yet.

        Args:
            num_filters: The number of output filters.
            layer_ind: The index of the layer (between 0 and num_layers-2).
                Note, that the last layer is fully-connected.
        """
        assert(not self._is_build)
        assert(layer_ind < self.num_gen_layers-1)
        self._conv_filters[layer_ind] = num_filters

    def get_kernel_size(self, layer_ind, layer_type='conv'):
        """Get the kernel size of a convolutional layer.

        Args:
            layer_ind: The index of the layer (between 0 and num_layers-2).
                Note, that the last layer is fully-connected.
            layer_type (default: 'conv'): Can be either 'conv' or 'pool'.
                decides whether the kernel size of the convolutional or pooling
                layer is considered.

        Returns:
            The kernel size as a tuple of integers.
        """
        assert(layer_ind < self.num_gen_layers-1)
        if layer_type == 'pool':
            return self._pool_kernels[layer_ind]
        return self._conv_kernels[layer_ind]

    def set_kernel_size(self, kernel_size, layer_ind, layer_type='conv'):
        """Set the kernel size of a convolutional layer.

        Note, this method can only be called for a network, that has not been
        build yet.

        Args:
            kernel_size: The new kernel size.
            layer_ind: The index of the layer (between 0 and num_layers-2).
                Note, that the last layer is fully-connected.
            layer_type (default: 'conv'): Can be either 'conv' or 'pool'.
                decides whether the kernel size of the convolutional or pooling
                layer is considered.
        """
        assert(not self._is_build)
        assert(layer_ind < self.num_gen_layers-1)
        if layer_type == 'pool':
            self._pool_kernels[layer_ind] = kernel_size
        self._conv_kernels[layer_ind] = kernel_size

    def build(self):
        """Build the network, such that we can use it to run training or
        inference.

        Note, this method doesn't really distinguish between training and
        inference mode. The built network, will be the same in both cases.
        However, there is a dedicated placeholder, that can be passed to tell
        the network, that it runs in training mode.
        """
        logger.info('Building CNN Classifier ...')

        self._is_build = True

        tf.reset_default_graph()
        self._graph = tf.Graph()
        with self._graph.as_default() as g:
            with g.name_scope(self._scope_name) as scope:
                self._build_datasets()

                d_in_size = np.prod(shared.data.in_shape)
                num_classes = shared.data.num_classes
                # We use placeholder_with_default to ensure, that we can either
                # load samples from the dataset, or feed random samples from
                # the user.
                self._t_inputs = tf.placeholder_with_default(
                    self._t_ds_inputs, shape=[None, d_in_size], name='inputs')
                self._t_labels = tf.placeholder_with_default(
                    self._t_ds_outputs, shape=[None, num_classes],
                    name='labels')
                # This tensor is used to distinguish between a network run in
                # training or inference mode.
                # The default option will be to run the network in inference
                # mode.
                self._t_mode = tf.placeholder_with_default(False, shape=[],
                                                           name='is_training')

                in_shape = shared.data.in_shape
                if np.size(in_shape) == 2:
                    in_shape = in_shape + [1]
                inputs_reshaped = tf.reshape(self._t_inputs, [-1] + in_shape)

                self._build_network(inputs_reshaped)

                # Compute loss.
                self._t_loss = tf.losses.softmax_cross_entropy(self._t_labels,
                                                               self._t_outputs)
                tf.summary.scalar('loss', self._t_loss)

                # Compute the accuracy of the network.
                correct_predictions = tf.equal(tf.argmax(self._t_outputs, 1),
                                               tf.argmax(self._t_labels, 1))
                self._t_accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                          tf.float32))
                tf.summary.scalar('accuracy', self._t_accuracy)

                self._t_summaries = tf.summary.merge_all()

                self._t_global_step = tf.get_variable('global_step', shape=(),
                    initializer=tf.constant_initializer(0), trainable=False)

            self._scope = scope

            logger.info('Building CNN Classifier ... Done')

    def train(self, num_iter=10000, batch_size=32, learning_rate=0.001, \
              momentum=0.9, val_interval=1000, val_bs=1000):
        """Train the network.

        The network is trained via a Momentum Optimizer.

        Note, if no validation set is available, the test set will be used.

        Args:
            num_iter: The number of training iterations.
            batch_size: The training batch size.
            learning_rate: See docs of "tf.train.MomentumOptimizer".
            momentum: See docs of "tf.train.MomentumOptimizer".
            val_interval: How often the training status should be validated.
            val_bs: The batch size of the validation set to use.
        """
        if not self._is_build:
            raise CustomException('Network has not been build yet.')
            
        logger.info('Training CNN Classifier ...')

        with self._graph.as_default() as g:
            #print([v.name for v in tf.trainable_variables()])

            summary_writer = tf.summary.FileWriter( \
                self._get_summary_dir('train_summary',
                    rm_existing=not self.continue_training), g)

            self._init_validation(val_bs)

            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)

            # We need to add the update_ops for the batchnorm moving averages
            # to the training steps. Otherwise, they won't be executed.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = optimizer.minimize(self._t_loss,
                    global_step=self._t_global_step)

            init_op = tf.global_variables_initializer()

            checkpoint_saver = tf.train.Saver(max_to_keep=5, \
                                              keep_checkpoint_every_n_hours=3)

        with tf.Session(graph=self._graph, config=self._get_config_proto()) \
                as sess:
            # Restore training if requested.
            iter_start, iter_end = self._init_training(sess, num_iter, init_op,
                self._t_global_step, checkpoint_saver)

            # Initialize training set.
            train_handle = sess.run(self._train_iter.string_handle())
            sess.run(self._train_iter.initializer,
                 feed_dict={
                    self._t_train_raw_in: shared.data.get_train_inputs(),
                    self._t_train_raw_out: 
                        shared.data.get_train_outputs(use_one_hot=True),
                    self._t_train_batch_size: batch_size})

            for i in range(iter_start, iter_end):
                if i % val_interval == 0:
                    checkpoint_saver.save(sess, os.path.join( \
                        self._checkpoint_dir, 'model'), global_step=i)

                    self._validate_training_process(sess, i)

                elif i % 100 == 0 and i > 0:
                    logger.info('Running training epoch: %d.' % i)

                _, summary = sess.run([train_step, self._t_summaries],
                    feed_dict={self._t_handle: train_handle,
                               self._t_mode: True})
                summary_writer.add_summary(summary, i)

            checkpoint_saver.save(sess, os.path.join( \
                self._checkpoint_dir, 'model'), global_step=iter_end)
            logger.info('Training ends after %d iterations.' % iter_end)

        summary_writer.close()
        self._val_summary_writer.close()

        logger.info('Training CNN Classifier ... Done')


    def test(self):
        """Evaluate the trained network using the whole test set."""
        if not self._is_build:
            raise CustomException('Network has not been build yet.')

        logger.info('Testing CNN Classifier ...')

        sess = self._get_inference_session()
        if sess is None:
            logger.error('Could not create session. Testing aborted.')

        test_ins = shared.data.get_test_inputs()
        test_outs = shared.data.get_test_outputs(use_one_hot=True)

        test_handle = sess.run(self._test_iter.string_handle())
        sess.run(self._test_iter.initializer,
             feed_dict={self._t_test_raw_in: test_ins,
                        self._t_test_raw_out: test_outs,
                        self._t_test_batch_size: shared.data.num_test_samples})

        ckpt_epoch = tf.train.global_step(sess, self._t_global_step)
        logger.info('The network has been trained for %d epochs.' % 
                    ckpt_epoch)

        acc, loss = sess.run([self._t_accuracy, self._t_loss],
            feed_dict={self._t_handle: test_handle})

        logger.info('Test Accuracy: %f' % acc)
        logger.info('Loss on test set: %f' % loss)

        if self.allow_plots:
            num_plots = 8

            # We have to reinitialize to change the batch size (seems to be
            # a cleaner solution than processing the whole validation set).
            sess.run(self._test_iter.initializer,
                feed_dict={self._t_test_raw_in: test_ins[:num_plots, :],
                           self._t_test_raw_out: test_outs[:num_plots, :],
                           self._t_test_batch_size: num_plots})
            [inps, lbls, preds]= sess.run(
                [self._t_ds_inputs, self._t_ds_outputs, self._t_output_probs],
                feed_dict={self._t_handle: test_handle})

            shared.data.plot_samples('Test Samples', inps, outputs=lbls,
                                     predictions=preds, interactive=True)

        logger.info('Testing CNN Classifier ... Done')

    def run(self, inputs):
        """Run the network with the given inputs.

        Args:
            inputs: Samples that align with the dataset (2D numpy array).

        Returns:
            The predicted 1-hot encoded labels (i.e., a 2D numpy array with
            rows encoding individual samples).
        """
        sess = self._get_inference_session()
        if sess is None:
            logger.error('Could not create session. Inference aborted.')

        [preds] = sess.run([self._t_output_probs],
                feed_dict={self._t_inputs: inputs})

        return preds

    def _init_validation(self, val_batch_size):
        """Initialize the validation process, that is called regularly during
        training.

        Args:
            val_batch_size: Size of the validation batch. Note, that every time
                            the validation is called, the same samples should
                            be used.
        """
        self._val_batch = shared.data.next_val_batch(val_batch_size,
                                                     use_one_hot=True)
        if self._val_batch is None:
            self._val_batch = shared.data.next_test_batch(val_batch_size,
                                                          use_one_hot=True)

        self._val_summary_writer = tf.summary.FileWriter( \
            self._get_summary_dir('val_summary',
                                  rm_existing=not self.continue_training),
            self._graph)

    def _validate_training_process(self, sess, epoch):
        """Validate the current training process on the validation batch. Note,
        that the validation uses the same graph and session as the training,
        but the training mode tensor (received as an input tensor of the
        network) is different.

        Args:
            sess: The current training session.
            epoch: The current training iteration.
        """
        logger.info('Epoch %d: validating training process ...' % epoch)
        
        if self.val_cpu_only:
            logger.warn('The option \'val_cpu_only\' is enabled, but not ' + \
                        'supported by this class. Option will be ignored.')

        val_handle = sess.run(self._val_iter.string_handle())
        sess.run(self._val_iter.initializer,
             feed_dict={self._t_val_raw_in: self._val_batch[0],
                        self._t_val_raw_out: self._val_batch[1],
                        self._t_val_batch_size: self._val_batch[0].shape[0]})

        loss, acc, summary = sess.run( \
                [self._t_loss, self._t_accuracy, self._t_summaries],
                feed_dict={self._t_handle: val_handle})

        logger.info('Validation Accuracy: %f' % acc)
        logger.info('Loss on validation batch: %f' % loss)

        self._val_summary_writer.add_summary(summary, epoch)
        self._val_summary_writer.flush()

        if self.allow_plots:
            num_plots = 4

            # We have to reinitialize to change the batch size (seems to be
            # a cleaner solution than processing the whole validation set).
            sess.run(self._val_iter.initializer,
                feed_dict={self._t_val_raw_in:
                               self._val_batch[0][:num_plots, :],
                           self._t_val_raw_out:
                               self._val_batch[1][:num_plots, :],
                           self._t_val_batch_size: num_plots})
            [inps, lbls, preds] = sess.run(
                [self._t_ds_inputs, self._t_ds_outputs, self._t_output_probs],
                feed_dict={self._t_handle: val_handle})

            shared.data.plot_samples('Validation Samples at Epoch %d' % epoch,
                                     inps, outputs=lbls, predictions=preds,
                                     interactive=True)

        logger.info('Epoch %d: validating training process ... Done' % epoch)

    def _compute_network_layout(self):
        """This method computes the strides, paddings, filter sizes and kernel
        sizes for convolutional and pooling layers.
        
        A detailed explanation of how the network is set up can be found in the
        docstring of the setter of the attribute "num_layers".
        """
        # Only consider convolutional layers.
        num_layers = self._num_layers - 1

        # Configs for convolutional layers.
        self._conv_pads = ['same'] * num_layers
        self._conv_kernels = [[5,5]] * num_layers
        self._conv_filters = list(map(lambda i : 32*2**i, range(num_layers)))
        self._conv_strides = [1] * num_layers

        # Configs for pooling layers.
        self._pool_pads = ['same'] * num_layers
        self._pool_kernels = [[2,2]] * num_layers
        self._pool_strides = [2] * num_layers
        
    def _build_network(self, inputs):
        """This network builds the actual network, consisting of several 2D
        convolutional layers and a final FC layer.
        
        Args:
            inputs: The tensor containing the inputs to the network (a 2D
                    image).
        """
        tf.summary.image('input', inputs)

        logger.info('Network input shape: [%s]' % (', '.join([str(e) for e in \
                                                   inputs.shape[1:]])))

        # Convolutional layers.
        last_layer = inputs
        self._t_hidden = []
        for i in range(self.num_layers-1):
            with tf.variable_scope('hidden_%d' % (i)):
                if self.use_batchnorm:
                    use_bn = False
                    is_training = None
                else:
                    use_bn = True
                    is_training = self._t_mode

                hidden = layers.conv2d(last_layer, self._conv_filters[i],
                    self._conv_kernels[i], strides=self._conv_strides[i],
                    padding=self._conv_pads[i], activation=tf.nn.relu,
                    pool_size=self._pool_kernels[i],
                    pool_strides=self._pool_strides[i],
                    pool_padding=self._pool_pads[i], use_bn=use_bn,
                    is_training=is_training)
                self._t_hidden.append(hidden)
                last_layer = hidden
                
                logger.info('Output shape after %d. convolutional layer: ' %
                            (i+1) + '[%s]' % (', '.join([str(e) for e in
                                              last_layer.shape[1:]])))

        curr_shape = int(np.prod(last_layer.shape[1:]))
        last_layer = tf.reshape(last_layer, [-1, curr_shape])

        # Final fully connected layer.
        with tf.variable_scope("output"):
            curr_shape = int(np.prod(last_layer.shape[1:]))
            last_layer = tf.reshape(last_layer, [-1, curr_shape])
            self._t_outputs = layers.fully_connected(last_layer,
                shared.data.num_classes, activation_fn=tf.nn.relu)
            
            # For visualizations.
            self._t_output_probs = tf.contrib.layers.softmax(self._t_outputs)
            tf.summary.histogram('output', self._t_output_probs)

if __name__ == '__main__':
    pass


