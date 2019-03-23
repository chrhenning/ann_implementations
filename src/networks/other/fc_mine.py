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
@title           :networks/other/fc_mine.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :09/26/2018
@version         :1.0
@python_version  :3.6.6

A simple fully connected network to test the Mutual Information Neural
Estimator (MINE), published in:
    Belghazi et al., MINE: Mutual Information Neural Estimation, 2018, arXix.
    https://arxiv.org/abs/1801.04062
"""
import tensorflow as tf
import numpy as np
import os

from networks.network_base_tf import NetworkBaseTF
import misc.shared_vars as shared
import layers.tf_layers as layers
from misc.custom_exceptions.custom_exception import CustomException

import logging
logger = logging.getLogger(shared.logging_name)

class FCMINE(NetworkBaseTF):
    """An implementation of a fully-connected neural network estimating the
    Mutual Information based on the MINE algorithm. The input dataset is
    expected to be a concatination of two random vectors whose mutual
    information shall be estimated. The output of the dataset has to be the
    true mutual information (only used for validation). Note, this network
    is only meant for testing the MINE approach.

    Note, that batch sizes need to be even numbers (see Algorithm 1 of the
    original paper), such that we can split the batch into a part drawn from
    the joint distribution and a part drawn from the product of the marginals
    (by mixing the two parts).

    Attributes (additional to base class):
        num_layers: The number of layers in the network.
    """
    _DEFAULT_NUM_LAYERS = 3

    def __init__(self, mode='train'):
        """
        Args:
            mode: The value of the attribute "mode".
        """
        super().__init__(mode)

        assert(np.prod(shared.data.out_shape) == 1)

        self._num_layers = FCMINE._DEFAULT_NUM_LAYERS

        self._compute_network_layout()

    @property
    def num_layers(self):
        """Getter for the attribute num_layers."""
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        """Setter for the attribute num_layers.

        Note, that this setter will overwrite all layer sizes, that might
        have been already set.

        Here is how the default network is set up.
        The default network will have 100 neurons in its first fully connected
        layer and 1 output neuron in the last layer. In between, the layer
        sizes shrink linearly.
        """
        if self._is_build:
            raise CustomException('The num_layers attribute can only be '
                                  + 'changed if the network has not been '
                                  + 'build yet.')
        self._num_layers = value

        self._compute_network_layout()

    def get_layer_size(self, layer_ind):
        """Get the layer size.

        Args:
            layer_ind: The index of the layer (between 0 and num_layers-1).
                Note, that the last layer always has a single output.

        Returns:
            The size of the layer.
        """
        assert(layer_ind < self.num_layers)
        return self._layer_sizes[layer_ind]

    def set_layer_size(self, layer_size, layer_ind):
        """Set the size of a hidden layer.

        Note, this method can only be called for a network, that has not been
        build yet.

        Args:
            layer_size: The new layer size.
            layer_ind: The index of the layer (between 0 and num_layers-2).
                Note, that the last layer always has a single output.
        """
        assert(not self._is_build)
        assert(layer_ind < self.num_layers-1)
        self._layer_sizes[layer_ind] = layer_size

    def build(self):
        """Build the network, such that we can use it to run training or
        inference.
        """
        logger.info('Building Fully-connected MI Estimator ...')

        self._is_build = True

        tf.reset_default_graph()
        self._graph = tf.Graph()
        with self._graph.as_default() as g:
            with g.name_scope(self._scope_name) as scope:
                self._build_datasets()

                d_in_size = np.prod(shared.data.in_shape)
                # We need to split each sample into the two random vectors
                # (coming from the random distributions whose MI should be
                # estimated).
                assert(d_in_size % 2 == 0)

                # We use placeholder_with_default to ensure, that we can either
                # load samples from the dataset, or feed random samples from
                # the user.
                self._t_inputs = tf.placeholder_with_default(
                    self._t_ds_inputs, shape=[None, d_in_size], name='inputs')
                # This tensor is used to distinguish between a network run in
                # training or inference mode.
                # The default option will be to run the network in inference
                # mode.
                self._t_mode = tf.placeholder_with_default(False, shape=[],
                                                           name='is_training')
                # This is not used, except for plotting it in tensorboard.
                self._t_real_mi = tf.placeholder_with_default(
                    tf.reduce_mean(self._t_ds_outputs), shape=[],
                    name='real_mi')
                # If we are running the network with samples, where the exact
                # MI is not known, then we can't plot the real MI.
                self._t_mi_known = tf.placeholder_with_default(False, shape=[],
                    name='real_mi_known')

                batch_size = tf.shape(self._t_inputs)[0]
                # Note, that we need to split the batch into two parts, one
                # being pairs of samples (sampled from the joint) and another
                # part that is mixed with the first part (sampled from
                # product of marginals).
                tf.Assert(tf.equal(tf.floormod(batch_size, 2), 0),
                          [batch_size])

                # Build joint and marginal batch.
                joint_batch = tf.slice(self._t_inputs, [0, 0],
                                       [batch_size // 2, d_in_size])

                # Extract X from first half of samples.
                X = tf.slice(self._t_inputs, [0, 0],
                             [batch_size // 2, d_in_size // 2])
                # Extract Z from second half.
                Z = tf.slice(self._t_inputs, [batch_size // 2, d_in_size // 2],
                             [batch_size // 2, d_in_size // 2])

                marginal_batch = tf.concat([X, Z], axis=1)

                # Build statistics network.
                self._t_joint_hidden, self._t_joint_outputs = \
                    self._statistics_network(joint_batch, reuse=False)
                self._t_marginal_hidden, self._t_marginal_outputs = \
                    self._statistics_network(marginal_batch, reuse=True)

                # Compute MI estimate.
                self._t_mi = tf.reduce_mean(self._t_joint_outputs) - \
                    tf.log(tf.reduce_mean(tf.exp(self._t_marginal_outputs)))

                tf.summary.scalar('MI_estimate', self._t_mi)
                tf.cond(self._t_mi_known, lambda: tf.summary.scalar('MI_real',
                    self._t_real_mi), lambda: 'None')

                # Compute loss.
                # We wanna maximize the estimate (supremum over function
                # space), thus minimize the negative estimate.
                self._t_loss = -self._t_mi

                self._t_summaries = tf.summary.merge_all()

                self._t_global_step = tf.get_variable('global_step', shape=(),
                    initializer=tf.constant_initializer(0), trainable=False)

            self._scope = scope
        
        logger.info('Building Fully-connected MI Estimator ... Done')

    def train(self, num_iter=10000, batch_size=256, learning_rate=0.0001, \
              beta1=0.5, beta2=0.999, val_interval=1000, val_bs=256):
        """Train the network.

        The network is trained via the Adam optimizer.

        Note, if no validation set is available, the test set will be used.

        Args:
            num_iter: The number of training iterations.
            batch_size: The training batch size.
            learning_rate: See docs of "tf.train.AdamOptimizer".
            beta1: See docs of "tf.train.AdamOptimizer".
            beta2: See docs of "tf.train.AdamOptimizer".
            val_interval: How often the training status should be validated.
            val_bs: The batch size of the validation set to use.
        """
        if not self._is_build:
            raise CustomException('Network has not been build yet.')
            
        logger.info('Training MI Estimator ...')
        
        with self._graph.as_default() as g:
            summary_writer = tf.summary.FileWriter( \
                self._get_summary_dir('train_summary',
                                      rm_existing=not self.continue_training),
                                      g)

            self._init_validation(val_bs)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=beta1, beta2=beta2)
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
                     feed_dict={self._t_train_raw_in:
                                     shared.data.get_train_inputs(),
                                self._t_train_raw_out:
                                     shared.data.get_train_outputs(),
                                self._t_train_batch_size: batch_size})

            for i in range(iter_start, iter_end):
                if i % val_interval == 0:
                    checkpoint_saver.save(sess, os.path.join( \
                        self._checkpoint_dir, 'model'), global_step=i)

                    self._validate_training_process(sess, i)

                #elif i % 100 == 0 and i > 0:
                #    logger.info('Running training epoch: %d.' % i)

                _, summary = sess.run( \
                    [train_step, self._t_summaries],
                    feed_dict={self._t_handle: train_handle,
                               self._t_mode: True,
                               self._t_mi_known: True})
                summary_writer.add_summary(summary, i)

            checkpoint_saver.save(sess, os.path.join( \
                self._checkpoint_dir, 'model'), global_step=iter_end)
            logger.info('Training ends after %d iterations.' % iter_end)

        summary_writer.close()
        self._val_summary_writer.close()

        logger.info('Training MI Estimator ... Done')

    def test(self):
        """Evaluate the trained network using the whole test set."""
        if not self._is_build:
            raise CustomException('Network has not been build yet.')

        logger.info('Testing MI Estimator ...')

        sess = self._get_inference_session()
        if sess is None:
            logger.error('Could not create session. Testing aborted.')

        test_ins = shared.data.get_test_inputs()
        test_outs = shared.data.get_test_outputs()

        test_handle = sess.run(self._test_iter.string_handle())
        sess.run(self._test_iter.initializer,
             feed_dict={self._t_test_raw_in: test_ins,
                        self._t_test_raw_out: test_outs,
                        self._t_test_batch_size: shared.data.num_test_samples})

        ckpt_epoch = tf.train.global_step(sess, self._t_global_step)
        logger.info('The network has been trained for %d epochs.' % 
                    ckpt_epoch)

        real_mi, estimated_mi = sess.run([self._t_real_mi, self._t_mi],
            feed_dict={self._t_handle: test_handle,
                       self._t_mi_known: True})

        logger.info('Real MI: %f' % real_mi)
        logger.info('Estimated MI on test set: %f' % estimated_mi)

        logger.info('Testing MI Estimator ... Done')

    def run(self, inputs):
        """Run the network with the given inputs.

        Args:
            inputs: Samples that align with the dataset (2D numpy array). Note,
                the size of the batch must be even.

        Returns:
            The estimated MI.
        """
        sess = self._get_inference_session()
        if sess is None:
            logger.error('Could not create session. Inference aborted.')

        [preds] = sess.run([self._t_mi],
                feed_dict={self._t_inputs: inputs})

        return preds

    def _compute_network_layout(self):
        """This network computes the number of units per layer, if the number
        of layers is reset.
        """
        self._layer_sizes = np.round(np.linspace(100, 1, self._num_layers)). \
            astype(np.int64).tolist()

    def _statistics_network(self, inputs, reuse=False):
        """ Build the statistics network.
        
        Args:
            inputs: The concatinated realization drawn from the two random
                    vectors.
            reuse (default: False): Whether the created variables can be
                  reused.

        Returns:
            [t_hidden, t_outputs]
            t_hidden is a list of hidden layer output tensors.
            t_outputs is the output tensor of the statistics network.
        """
        # Hidden layers.
        last_layer = inputs
        t_hidden = []
        for i in range(self.num_layers-1):
            with tf.variable_scope('hidden_%d' % (i), reuse=reuse):
                hidden = layers.fully_connected(last_layer, 
                    self._layer_sizes[i], activation_fn=tf.nn.elu)
                t_hidden.append(hidden)
                last_layer = hidden

        # Output layer.
        with tf.variable_scope("output", reuse=reuse):
            t_outputs = layers.fully_connected(last_layer, 1,
                                               activation_fn=None)

        return [t_hidden, t_outputs]

    def _init_validation(self, val_batch_size):
        """Initialize the validation process, that runs in parallel to the
        training.

        Args:
            val_batch_size: Size of the validation batch.
        """
        self._val_batch = shared.data.next_val_batch(val_batch_size)
        if self._val_batch is None:
            self._val_batch = shared.data.next_test_batch(val_batch_size)

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

        mi_estimate, mi_real, summary = sess.run( \
                [self._t_mi, self._t_real_mi, self._t_summaries],
                feed_dict={self._t_handle: val_handle,
                           self._t_mi_known: True,})

        logger.info('Real MI: %f' % mi_real)
        logger.info('Estimated MI on validation batch: %f' % mi_estimate)

        self._val_summary_writer.add_summary(summary, epoch)
        self._val_summary_writer.flush()

        logger.info('Epoch %d: validating training process ... Done' % epoch)

if __name__ == '__main__':
    pass


