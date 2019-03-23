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
@title           :networks/autoencoder/simple_ae.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/09/2018
@version         :1.0
@python_version  :3.6.6

This network is thought as a simple example implementation for a network in
this framework.

This implementation highlights, how to do a clearcut between training and
validation network, in case this might be necessary.
Note, that some network elements (such as BatchNorm) behave differently in
training versus inference mode. One could solve this by having an extra
placeholder, that inputs the network mode to the network. Here, we do this
by having a separate (second) network.

What is the advantage of this implementation? Assume, that the validation is
time consuming. One could run the training on the GPU, while having a lot of
computation available on its CPU. This can be exploited by running the
validation in a separate thread, that creates a session, that doesn't utilize
the GPU. Hence, training is not slowed down, whereas evaluation runs constantly
in parallel.

You may start two Tensorboard sessions to follow the training and validation
process.
"""

import numpy as np
import tensorflow as tf
import os
import threading

from networks.network_base_tf import NetworkBaseTF
import misc.shared_vars as shared
from misc.custom_exceptions.custom_exception import CustomException
import layers.tf_layers as layers
import misc.visualizations.plotting_data as dplt

import logging
logger = logging.getLogger(shared.logging_name)

class SimpleAE(NetworkBaseTF):
    """A simple multilayer autoencoder with fully-connected layers.

    Attributes (additional to base class):
        num_layers: The number of layers in the network, including input and
            output layer.
    """
    class ValThread(threading.Thread):
        """This subclass is needed to run the validation in a separate thread.

        Hence, it's whole purpose is to create a validation session, restore
        the latest weights and sweep through the validation batch.
        """
        def __init__(self, train_net, epoch):
            """Create a new validation thread.

            Args:
                train_net: An object of the class SimpleAE, that runs the
                    training, that should be evaluated by this thread.
                epoch: The global step of the training, that should be assessed
                    by this thread.
            """
            threading.Thread.__init__(self)
            self._train_net = train_net
            self._epoch = epoch

        def run(self):
            """This method restores a training checkpoint. If the latest
            checkpoint doesn't correspond to the desired "epoch", this method
            will log an error and do nothing. Otherwise, it will process the
            validation batch through the validation network.
            
            Note, the validation network was constructed in the outer class.
            """
            tnet = self._train_net
            vnet = tnet._val_net
            epoch = self._epoch

            ckpt_path = tnet._latest_checkpoint_path()
            if not ckpt_path:
                logger.error('No validation at epoch %d.' % epoch)
                return

            logger.info('Epoch %d: validating training process ...' % epoch)

            sess_config = tnet._get_config_proto(cpu_only=tnet.val_cpu_only)
            with tf.Session(graph=vnet._graph, \
                            config=sess_config) as sess:
                tnet._val_saver.restore(sess, ckpt_path)
                ckpt_epoch = tf.train.global_step(sess, vnet._t_global_step)
                
                if ckpt_epoch != epoch:
                    logger.error('Could not load correct checkpoint file ' +
                                 'for validation at epoch %d.' % epoch)
                    return

                val_handle = sess.run(vnet._val_iter.string_handle())
                sess.run(vnet._val_iter.initializer,
                     feed_dict={vnet._t_val_raw_in: tnet._val_batch[0],
                                vnet._t_val_raw_out: tnet._val_batch[1],
                                vnet._t_val_batch_size:
                                    tnet._val_batch[0].shape[0]})

                [inputs, labels, reconstruction, loss, summary] = sess.run(
                    [vnet._t_ds_inputs, vnet._t_ds_outputs,
                     vnet._t_outputs, vnet._t_loss, vnet._t_summaries],
                    feed_dict={vnet._t_handle: val_handle})
                tnet._val_summary_writer.add_summary(summary, epoch)
                tnet._val_summary_writer.flush()
                
                logger.info('The current validation loss is: %f' % loss)

            # FIXME Have to use multiprocessing to plot within this thread:
            #https://stackoverflow.com/questions/34764535/why-cant-matplotlib-plot-in-a-different-thread

            # Though, it seems to work for the inline backend from iPython:
            import matplotlib
            if tnet.allow_plots and 'inline' in matplotlib.get_backend():
                # Plot the first 3 validation samples.
                img_dataset, _ = shared.data.is_image_dataset()
                if img_dataset:
                    np = min(3, inputs.shape[0])
    
                    dplt.plot_ae_images('Validation Samples',
                                        inputs[:np, :],
                                        reconstruction[:np, :],
                                        sample_outputs=labels[:np, :],
                                        interactive=True,
                                        figsize=(10, 3))


            logger.info('Epoch %d: validating training process ... Done' % epoch)

    _DEFAULT_NUM_HLAYERS = 3
    _DEFAULT_SIZE_HLAYERS = [100, 30, 100]
    
    def __init__(self, mode='train'):
        """
        Args:
            mode: The value of the attribute "mode".
        """
        super().__init__(mode)

        # This network is not recurrent.
        if shared.data.sequence:
            raise CustomException("The network doesn't support sequence data.")

        self._num_layers = SimpleAE._DEFAULT_NUM_HLAYERS + 2
        in_size = np.prod(shared.data.in_shape)
        self._layer_sizes = [in_size] +  SimpleAE._DEFAULT_SIZE_HLAYERS + \
            [in_size]

    @property
    def num_layers(self):
        """Getter for the attribute num_layers."""
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value):
        """Setter for the attribute num_layers.

        Note, this method can only be called for a network, that has not been
        build yet.

        Note, that an autoencoder has typically a bottleneck. So one should
        prefer odd numbers of layers.

        The hidden layer sizes are all set to 100 after this method has been
        called. Ensure that you set them to meaningful values afterwards.

        Args:
            value: The number of layers in this network (incl. in- and output 
            layer).
        """
        if self._is_build:
            raise CustomException('The number of layers in a network can only'
                                  + ' be changed if the network has not been '
                                  + ' build yet.')

        if value < 2:
            raise CustomException('A network needs at least 2 layers ' +
                                  '(input and output)')

        self._num_layers = value
        
        # Initialize the hidden layer sizes.
        in_size = self._layer_sizes[0]
        self._layer_sizes = [100] * value
        self._layer_sizes[0] = in_size
        self._layer_sizes[-1] = in_size

    def get_layer_size(self, layer_ind):
        """Get the size of any layer in this network.

        Args:
            layer_ind: The index of the layer (between 0 and num_layers-1).

        Returns:
            The size of the requested layer.
        """
        return self._layer_sizes[layer_ind]

    def set_hidden_layer_size(self, layer_ind, layer_size):
        """Set the size of an hidden layer.

        Note, the index of the first hidden layer is 1 (the last has index
        num_layers-2)!

        The same restrictions as for the num_layers setter apply.

        Args:
            layer_ind: The index of the hidden layer.
            layer_size: The new size of the layer.
        """
        if self._is_build:
            raise CustomException('The hidden layer size can only be changed '
                                  + 'if the network has not been build yet.')

        if layer_ind < 1 or layer_ind > self.num_layers-2:
            raise CustomException('Hidden layers have an index between 1 and '
                                  + str(self.num_layers-2) + '.')

        self._layer_sizes[layer_ind] = layer_size

    def build(self):
        """Build the network, such that we can use it to run training or
        inference."""
        self._is_build = True

        msg = 'Building a ' + str(self.num_layers) + '-layer fully-' + \
              'connected autoencoder for the dataset: ' + \
              shared.data.get_identifier()
        logger.info(msg)

        # Note, that we need to remember the graph, in case we want to
        # construct multiple instances of this class (or multiple tensorflow
        # graphs in general). Because in this case, we can't just refer to the
        # default graph.
        tf.reset_default_graph()
        self._graph = tf.Graph()
        with self._graph.as_default() as g:
            # Note, that we have to use name_scope, not variable_scope, as we
            # do want to keep the same variable names across instances of this
            # class (otherwise, we couldn't restore the graph as simply from a
            # checkpoint). However, we want to define a named scope for
            # Tensorboard visualizations, that's why we use name_scope. Note,
            # that this will only affect the readibility within a Tensorboard
            # session. We cannot write multiple graphs within the same 
            # Tensorboard session.
            with g.name_scope(self._scope_name) as scope:
                self._build_datasets()

                # Network Inputs.
                self._t_inputs = tf.placeholder_with_default(
                    self._t_ds_inputs, shape=[None, self._layer_sizes[0]],
                    name='inputs')
                self._t_learning_rate = tf.placeholder(tf.float32, shape=[])

                batch_size = tf.shape(self._t_inputs)[0]

                last_layer = self._t_inputs

                # Hidden layers.
                self._t_hidden = []
                for i in range(1, self.num_layers-1):
                    with tf.variable_scope('hidden_%d' % (i)):
                        hidden = layers.fully_connected(last_layer, \
                            self._layer_sizes[i], activation_fn=tf.nn.relu)
                        self._t_hidden.append(hidden)
                        last_layer = hidden

                # Input reconstructions.
                with tf.variable_scope("output"):
                    self._t_outputs = layers.fully_connected(last_layer, \
                        int(self._layer_sizes[-1]), activation_fn=tf.nn.relu)

                img_dataset, _ = shared.data.is_image_dataset()
                if img_dataset:
                    tf.summary.image('reconstruction', \
                        tf.reshape(self._t_outputs, \
                                   [-1] + shared.data.in_shape))

                # Compute loss: Reconstruction Error
                diff = self._t_inputs - self._t_outputs
                reconstruction_err = tf.nn.l2_loss(diff) / \
                    float(self._layer_sizes[0])
                reconstruction_err /= tf.to_float(batch_size)
                tf.losses.add_loss(reconstruction_err)

                self._t_loss = tf.losses.get_total_loss()
                tf.summary.scalar('loss', self._t_loss)

                self._t_summaries = tf.summary.merge_all()

                self._t_global_step = tf.get_variable('global_step', shape=(),
                    initializer=tf.constant_initializer(0), trainable=False)

        self._scope = scope

    def train(self, num_iter=10000, batch_size=32, init_lr=20, \
              lr_decay_interval=1000, val_interval=1000, val_bs=1000):
        """Train the network.

        The network is trained via gradient descent with decreasing learning
        rate.

        Note, if no validation set is available, the test set will be used.

        Args:
            num_iter: The number of training iterations.
            batch_size: The training batch size.
            init_lr: The initial learning rate.
            lr_decay_interval: After how many iterations the learning rate
                should be halved. If None, no weight decay is applied.
            val_interval: How often the training status should be validated.
            val_bs: The batch size of the validation set to use.
        """
        if not self.is_training():
            raise CustomException('Method can only be called in training ' + \
                                  'mode.')
        if not self._is_build:
            raise CustomException('Network has not been build yet.')

        logger.info('Training autoencoder ...')

        # Learning Rate
        lr = init_lr

        with self._graph.as_default() as g:
            #print([v.name for v in tf.trainable_variables()])

            summary_writer = tf.summary.FileWriter( \
                self._get_summary_dir('train_summary'), g)

            checkpoint_saver = tf.train.Saver(max_to_keep=5, \
                                              keep_checkpoint_every_n_hours=3)

            init_op = tf.global_variables_initializer()

            train_step = tf.train.GradientDescentOptimizer( \
                learning_rate=self._t_learning_rate).minimize(self._t_loss,
                    global_step=self._t_global_step)

        self._build_validation_graph(val_bs)

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
                        shared.data.get_train_outputs(),
                    self._t_train_batch_size: batch_size})

            for i in range(iter_start, iter_end):
                if i % val_interval == 0:
                    checkpoint_saver.save(sess, os.path.join( \
                        self._checkpoint_dir, 'model'), global_step=i)

                    self._validate_training(i)

                _, summary = sess.run([train_step, self._t_summaries],
                    feed_dict={self._t_handle: train_handle,
                               self._t_learning_rate: lr})
                summary_writer.add_summary(summary, i)

                # Exponential weight decay.
                if not lr_decay_interval is None and \
                        i > 0 and i % lr_decay_interval == 0:
                    lr /= 2
                    logger.info('Epoch %d: learning rate decayed to: %f' % \
                                (i, lr))

            checkpoint_saver.save(sess, os.path.join( \
                self._checkpoint_dir, 'model'), global_step=iter_end)
            logger.info('Training ends after %d iterations.' % iter_end)

        summary_writer.close()

        # Wait until all validation threads are done (so that we don't close
        # the summary writer too early).
        [t.join() for t in self._val_threads]
        self._val_summary_writer.close()

        logger.info('Training autoencoder ... Done')

    def test(self):
        """Evaluate the trained network using the whole test set.
        
        At the moment, this method simply computes the loss on the test set.
        """
        if self.is_training():
            raise CustomException('Method can only be called in inference ' + \
                                  'mode.')
        if not self._is_build:
            raise CustomException('Network has not been build yet.')

        logger.info('Testing autoencoder ...')

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

        [loss] = sess.run([self._t_loss], 
            feed_dict={self._t_handle: test_handle})

        logger.info('Loss on test dataset is: %f' % loss)

        # Plot example test images.
        img_dataset, _ = shared.data.is_image_dataset()
        if self.allow_plots and img_dataset:
            num_plots = 6

            # We have to reinitialize to change the batch size (seems to be
            # a cleaner solution than processing the whole validation set).
            sess.run(self._test_iter.initializer,
                feed_dict={self._t_test_raw_in: test_ins[:num_plots, :],
                           self._t_test_raw_out: test_outs[:num_plots, :],
                           self._t_test_batch_size: num_plots})
            [inputs, reconstructions, labels] = sess.run(
                [self._t_ds_inputs, self._t_outputs, self._t_ds_outputs], 
                feed_dict={self._t_handle: test_handle})

            dplt.plot_ae_images('Reconstructed Test Samples',
                                inputs, reconstructions, sample_outputs=labels,
                                interactive=True)

        logger.info('Testing autoencoder ... Done')

    def run(self, inputs):
        """Run the network with the given inputs.

        Args:
            inputs: Samples that align with the dataset (2D numpy array).

        Returns:
            The outputs of the network as 2D numpy array.
        """
        if self.is_training():
            raise CustomException('Method can only be called in inference ' + \
                                  'mode.')
        if not self._is_build:
            raise CustomException('Network has not been build yet.')

        sess = self._get_inference_session()
        if sess is None:
            logger.error('Could not create session. Inference aborted.')

        [outputs] = sess.run([self._t_outputs],
            feed_dict={self._t_inputs: inputs})

        return outputs

    def _build_validation_graph(self, val_batch_size):
        """Create a second instance of this class, that can be used as a
        validation network. Hence, this instance will be build in "inference"
        mode.

        Here is an example on how to get two Tensorflow graphs running in
        parallel:
            https://tinyurl.com/y9sx9qpa

        Note, this method will also generate the validation dataset.

        Args:
            val_batch_size: Size of validation batch.
        """
        self._val_batch = shared.data.next_val_batch(val_batch_size)
        if self._val_batch is None:
            self._val_batch = shared.data.next_test_batch(val_batch_size)

        self._val_threads = []

        self._val_net = SimpleAE(mode='inference')
        self._val_net._num_layers = self._num_layers
        self._val_net._layer_sizes = self._layer_sizes
        self._val_net.set_scope_name('val_scope')

        self._val_net.build()

        with self._val_net._graph.as_default() as g:
            self._val_saver = tf.train.Saver()

            # As we have a different graph, we also need a different
            # Tensorboard directory (a summary writer can only handle one
            # graph).
            self._val_summary_writer = tf.summary.FileWriter( \
                self._get_summary_dir('val_summary'), g)

    def _validate_training(self, epoch):
        """Run the evaluation of the training progress.

        This method will run the validation in a separate thread.

        Args:
            epoch: The current global training step, that should be evaluated.
        """
        val_thread = SimpleAE.ValThread(self, epoch)
        val_thread.start()
        
        self._val_threads.append(val_thread)

if __name__ == '__main__':
    pass


