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
@title           :networks/gan/dcgan.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/15/2018
@version         :1.0
@python_version  :3.6.6

This is an implementation of a DCGAN. DCGAN stands for Deep Convolutional GAN
and is an architecture that was proposed in this paper:
    https://arxiv.org/abs/1511.06434
    
Note, this implementation is thought as an example implementation of a GAN. We
do not claim, that our architecture mimics precisely the one proposed in the
paper.

Here are the architecture guidelines, that they propose in the paper:
 - Replace any pooling layers with strided convolutions (discriminator) and
   fractional-strided convolutions (generator).
 - Use batchnorm in both the generator and the discriminator.
 - Remove fully connected hidden layers for deeper architectures.
 - Use ReLU activation in generator for all layers except for the output, which
   uses Tanh.
 - Use LeakyReLU activation in the discriminator for all layers.

Note, the implementation doesn't really care about the "mode" attribute.
Instead, there is a special placeholder, that allows to distinguish between the
training and testing architecture.
"""

import numpy as np
import tensorflow as tf
import os

from networks.network_base_tf import NetworkBaseTF
import misc.shared_vars as shared
from misc.custom_exceptions.custom_exception import CustomException
import layers.tf_layers as layers
import misc.visualizations.plotting_data as dplt

import logging
logger = logging.getLogger(shared.logging_name)

class DCGAN(NetworkBaseTF):
    """A GAN implementation that uses convolutional layers in the generator
    and transpose convolutional layer in the discriminator.

    Attributes (additional to base class):
        num_gen_layers: Number of layers in the generator. The generator always
            consists of one fully-connected layer, that brings the latent input
            into a proper shape, followed by an arbitrary number of traspose
            convolutional layers.
        latent_size: The size of the generator input.
        num_dis_layers: The number of layers in the discriminator network. The
            discriminator network always consists of strided convolutional
            layers (no pooling) plus a final fully connected layer with a
            single output.
        use_biases: Whether layers can use bias terms (this applies to all
            layers, i.e., fully-connected, convolutional and transpose
            convolutional).
    """
    _DEFAULT_LATENT_SIZE = 100
    _DEFAULT_NUM_GLAYERS = 4
    _DEFAULT_NUM_DLAYERS = 4

    def __init__(self, mode='train'):
        """
        Args:
            mode: The value of the attribute "mode".
        """
        super().__init__(mode)

        # This network is not recurrent.
        if shared.data.sequence:
            raise CustomException("The network doesn't support sequence data.")

        # This is a GAN for images.
        img_dataset, _ = shared.data.is_image_dataset()
        if not img_dataset:
            raise CustomException("The network doesn't support datasets " + \
                                  "that do not have images as inputs.")

        self._latent_size = DCGAN._DEFAULT_LATENT_SIZE
        self._num_glayers = DCGAN._DEFAULT_NUM_GLAYERS
        self._compute_generator_layout()

        self._num_dlayers = DCGAN._DEFAULT_NUM_DLAYERS
        self._compute_discriminator_layout()

        self._use_biases = True

    @property
    def num_gen_layers(self):
        """Getter for the attribute num_gen_layers."""
        return self._num_glayers

    @num_gen_layers.setter
    def num_gen_layers(self, value):
        """Setter for the attribute num_gen_layers.

        Note, this method can only be called for a network, that has not been
        build yet.

        The first layer will be considered to be a fully-connected layer. All
        following layers are transpose convolutional layers.

        Note, that this method will override all kernel sizes and filters that
        might have been set for the generator already.

        How do we initialize an arbitrary deep generator network?

        Assume the dataset has input images of the shape [W,H,C]. This shape
        should be the output of the final transpose convolutional layer.

        The number of filters in each transpose convolutional (TC) layer is set
        the following way:
            The last one has C filters. The one before has 128. The one before
            this one has 256, ...
        Note, that we also assign a number of filters to the initial
        fully-connected layer in this way (as this one also outputs data via
        many channels).

        The usual configuration is, that each TC layer should roughly double
        its input size. If the output shape is already smaller than 4 (for
        either width or height), then the stride is set to 1 (and padding
        set to 'SAME', such that the size doesn't change).
        In all other cases, the stride is set to 2. If an output size is
        even, then the padding will be set to 'SAME'. If it is odd, the padding
        is going to be 'VALID' with a fixed kernel size of 3. This comes out
        of the equation:
            out_size = s*(in_size-1) + k - 2p
        where s - stride, k - kernel size, p - padding.
        Note, that valid padding means p = 0, such that we have (s=2, k=3):
            out_size = 2*(in_size-1) + 3 = 2 * in_size + 1
        Hence, we set the in_size to out_size // 2.

        A special case is if the parity of width and height is different. Then
        the padding is set to valid and the even size gets a kernel size of 2
        assigned, the odd one a kernel size of 3. This can again be verified
        with the above equations.

        Given our network construction above, we can only set the kernel size
        for layers that have 'SAME' padding.

        The default kernel size for the case of 'same' padding is 5x5.

        Args:
            value: The number of layers in the generator network.
        """
        if self._is_build:
            raise CustomException('The number of layers in a network can only'
                                  + ' be changed if the network has not been '
                                  + ' build yet.')

        if value < 1:
            raise CustomException('A generator needs at least 1 layer.')

        self._num_glayers = value
        self._compute_generator_layout()

    @property
    def latent_size(self):
        """Getter for the attribute latent_size."""
        return self._latent_size

    @latent_size.setter
    def latent_size(self, value):
        """Setter for the attribute latent_size.

        Note, this method can only be called for a network, that has not been
        build yet.
        """
        if self._is_build:
            raise CustomException('The latent size can only be changed if the '
                                  + 'network has not been build yet.')

        self._latent_size = value

    @property
    def num_dis_layers(self):
        """Getter for the attribute num_dis_layers."""
        return self._num_dlayers

    @num_dis_layers.setter
    def num_dis_layers(self, value):
        """Setter for the attribute num_dis_layers.

        Note, this method can only be called for a network, that has not been
        build yet.

        Note, that this method will override all kernel sizes and filters that
        might have been set for the discriminator already.

        How do we construct a discriminator network with a certain depth?

        If N is the number of layers in the discriminator. Then There will be
        N-1 convolutional layers followed by a single fully-connected layer.
        The fully connected layer has a single output. The convolutional layers
        are meant to downsample the input. Therefore, they are strided, usually
        with stride 2 (to half the input). If a layer input is smaller than 4
        (in any dimension), then the stride is reduced to 1.
        All layers use 'same' padding, such that the output size can always
        be computed as follows (irrespective of the chosen kernel size):
            out_size = ceil(in_size / 2)

        Filter sizes start with 64 and double with every layer.

        The predefined kernel size is 5x5.
        """
        if self._is_build:
            raise CustomException('The number of layers in a network can only'
                                  + ' be changed if the network has not been '
                                  + ' build yet.')

        self._num_dlayers = value
        self._compute_discriminator_layout()

    @property
    def use_biases(self):
        """Getter for the attribute use_biases."""
        return self._use_biases

    @use_biases.setter
    def use_biases(self, value):
        """Setter for the attribute use_biases."""
        if self._is_build:
           raise CustomException('The use_biases attribute can only be '
                                  + 'changed if the network has not been '
                                  + 'build yet.')
        self._use_biases = value

    def get_num_filters_gen(self, layer_ind):
        """Get the number output filter maps of a layer belonging to the
        generator.

        Args:
            layer_ind: The index of the layer (between 0 and num_gen_layers-1).
                Note, that the first layer also has output channels.

        Returns:
            The number of filters of the requested layer.
        """
        return self._gen_filters[layer_ind]

    def set_num_filters_gen(self, num_filters, layer_ind):
        """Set the number output filter maps of a layer belonging to the
        generator.

        Note, this method can only be called for a network, that has not been
        build yet.

        Args:
            num_filters: The number of output filters.
            layer_ind: The index of the layer (between 0 and num_gen_layers-2).
                Note, that the first layer also has output channels. The last
                layer its number of filters is predefined by the number of color
                channels.
        """
        assert(not self._is_build)
        assert(layer_ind < self.num_gen_layers-1)
        self._gen_filters[layer_ind] = num_filters

    def get_kernel_size_gen(self, layer_ind):
        """Get the kernel size of a transpose convolutional layer belonging
        to the generator.

        Args:
            layer_ind: The index of the layer (between 1 and num_gen_layers-1).
                Note, that the first layer is not a deconv layer.

        Returns:
            The kernel size as a tuple of integers.
        """
        assert (layer_ind > 0)
        return self._gen_kernels[layer_ind - 1]

    def set_kernel_size_gen(self, kernel_size, layer_ind):
        """Set the kernel size of a transpose convolutional layer belonging
        to the generator.

        Note, this method can only be called for a network, that has not been
        build yet.

        Args:
            kernel_size: The new kernel size. Note, that a kernel size can
                only be assigned to layers that use 'SAME' padding.
            layer_ind: The index of the layer (between 1 and num_gen_layers-1).
        """
        assert(not self._is_build)
        assert (layer_ind > 0)
        if self._gen_pads[layer_ind-1] != 'SAME':
            logger.warn('Could not set given kernel size, because the layer '
                        + 'does not use \'SAME\' padding.')
            return
        self._gen_kernels[layer_ind - 1] = kernel_size

    def get_padding_gen(self, layer_ind):
        """Get the padding of a transpose convolutional layer belonging to
        the generator.

        Args:
            layer_ind: The index of the layer (between 1 and num_gen_layers-1).
                Note, that the first layer is not a deconv layer.

        Returns:
            The padding of the layer.
        """
        assert (layer_ind > 0)
        return self._gen_pads[layer_ind - 1]

    def get_stride_gen(self, layer_ind):
        """Get the stride of a transpose convolutional layer belonging to the
        generator.

        Args:
            layer_ind: The index of the layer (between 1 and num_gen_layers-1).
                Note, that the first layer is not a deconv layer.

        Returns:
            The stride of the layer.
        """
        assert (layer_ind > 0)
        return self._gen_strides[layer_ind - 1]

    def get_num_filters_dis(self, layer_ind):
        """Get the number output filter maps of the convolutional layers
        belonging to the discrimintator.

        Args:
            layer_ind: The index of the layer (between 0 and num_gen_layers-2).
                Note, that the last layer (fully-connected) has no output
                channels.

        Returns:
            The number of filters of the requested layer.
        """
        assert (layer_ind < self.num_dis_layers-1)
        return self._dis_filters[layer_ind]

    def set_num_filters_dis(self, num_filters, layer_ind):
        """Set the number output filter maps of the convolutional layers
        belonging to the discrimintator.

        Note, this method can only be called for a network, that has not been
        build yet.

        Args:
            num_filters: The number of output filters.
            layer_ind: The index of the layer (between 0 and num_gen_layers-2).
                Note, that the last layer (fully-connected) has no output
                channels.
        """
        assert(not self._is_build)
        assert(layer_ind < self.num_dis_layers-1)
        self._dis_filters[layer_ind] = num_filters

    def get_kernel_size_dis(self, layer_ind):
        """Get the kernel size of the convolutional layers belonging to the
        discrimintator.

        Args:
            layer_ind: The index of the layer (between 0 and num_gen_layers-2).
                Note, that the last layer is fully-connected.

        Returns:
            The kernel size as a tuple of integers.
        """
        assert (layer_ind < self.num_dis_layers-1)
        return self._dis_kernels[layer_ind]

    def set_kernel_size_dis(self, kernel_size, layer_ind):
        """Set the kernel size of the convolutional layers belonging to the
        discrimintator.

        Note, this method can only be called for a network, that has not been
        build yet.

        Args:
            kernel_size: The new kernel size.
            layer_ind: The index of the layer (between 0 and num_gen_layers-2).
                Note, that the last layer is fully-connected.
        """
        assert(not self._is_build)
        assert (layer_ind < self.num_dis_layers-1)
        self._dis_kernels[layer_ind] = kernel_size

    def get_padding_dis(self, layer_ind):
        """Get the padding of the convolutional layers belonging to the
        discrimintator.

        Args:
            layer_ind: The index of the layer (between 0 and num_gen_layers-2).
                Note, that the last layer is fully-connected.

        Returns:
            The padding of the layer.
        """
        assert (layer_ind < self.num_dis_layers-1)
        return self._dis_pads[layer_ind]

    def get_stride_dis(self, layer_ind):
        """Get the stride of the convolutional layers belonging to the
        discrimintator.

        Args:
            layer_ind: The index of the layer (between 0 and num_gen_layers-2).
                Note, that the last layer is fully-connected.

        Returns:
            The stride of the layer.
        """
        assert (layer_ind < self.num_dis_layers-1)
        return self._dis_strides[layer_ind]

    def build(self):
        """Build the network, such that we can use it to run training or
        inference.

        Note, this method doesn't really distinguish between training and
        inference mode. The built network, will be the same in both cases.
        However, there is a dedicated placeholder, that can be passed to tell
        the network, that it runs in training mode.
        """
        self._is_build = True

        tf.reset_default_graph()
        self._graph = tf.Graph()
        with self._graph.as_default() as g:
            with g.name_scope(self._scope_name) as scope:
                self._build_datasets()

                d_in_size = np.prod(shared.data.in_shape)
                self._g_inputs = tf.placeholder(tf.float32, \
                    shape=[None, self._latent_size], name='gen_inputs')
                self._d_inputs = tf.placeholder_with_default( \
                    self._t_ds_inputs, shape=[None, d_in_size],
                    name='dis_inputs')
                # This tensor is used to distinguish between a network run in
                # training or inference mode.
                # The default option will be to run the network in inference
                # mode.
                self._t_mode = tf.placeholder_with_default(False, shape=[],
                                                           name='is_training')

                d_inputs_reshaped = tf.reshape(self._d_inputs,
                                               [-1] + shared.data.in_shape)

                # We should initialize all kernel weights using this
                # initializer.
                self._initializer = tf.truncated_normal_initializer(
                        stddev=0.02)

                with tf.variable_scope('gen'):
                    self._build_generator()

                tf.summary.image('real', d_inputs_reshaped)

                with tf.variable_scope('dis'):
                    self._d_hidden_fake, self._d_outputs_fake = \
                        self._build_discriminator(self._g_outputs,
                                                  reuse=False)
    
                    self._d_hidden_real, self._d_outputs_real = \
                        self._build_discriminator(d_inputs_reshaped,
                                                  reuse=True)

                # Collect variables, that belong either to generator or the
                # discriminator.
                self._g_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
                self._d_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
                #print([v.name for v in self._g_vars])
                #print([v.name for v in self._d_vars])

                # Subclasses don't have to set this tensor.
                self._t_accuracy = None

                # Loss functions
                self._compute_loss()

                tf.summary.scalar('gen_loss', self._g_loss)
                tf.summary.scalar('dis_loss', self._d_loss)
                if self._t_accuracy is not None:
                    tf.summary.scalar('dis_accuracy', self._t_accuracy)

                self._t_summaries = tf.summary.merge_all()

                self._t_global_step = tf.get_variable('global_step', shape=(),
                    initializer=tf.constant_initializer(0), trainable=False)

            self._scope = scope

    def train(self, num_iter=10000, batch_size=128, learning_rate=0.0002, \
              beta1=0.5, beta2=0.999, val_interval=1000, val_bs=1000):
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
            
        logger.info('Training DCGAN ...')

        with self._graph.as_default() as g:
            #print([v.name for v in tf.trainable_variables()])

            summary_writer = tf.summary.FileWriter( \
                self._get_summary_dir('train_summary',
                                      rm_existing=not self.continue_training),
                                      g)

            self._init_validation(val_bs)
            
            # TODO Such a dictionary should be part of the arguments of this
            # method to allow for easier choices of the used optimizer.
            op_params = {'learning_rate': learning_rate, 'beta1': beta1,
                         'beta2': beta2}
            gen_optimizer, dis_optimizer = self._get_optimizers(op_params)

            # We need to add the update_ops for the batchnorm moving averages
            # to the training steps. Otherwise, they won't be executed.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                gen_train_step = gen_optimizer.minimize(self._g_loss,
                    global_step=self._t_global_step, var_list=self._g_vars)
                dis_train_step = dis_optimizer.minimize(self._d_loss,
                    var_list=self._d_vars)

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

                elif i % 100 == 0 and i > 0:
                    logger.info('Running training epoch: %d.' % i)

                _, _, summary = sess.run( \
                    [gen_train_step, dis_train_step, self._t_summaries],
                    feed_dict={self._g_inputs: self.sample_latent(batch_size),
                               self._t_handle: train_handle,
                               self._t_mode: True})
                summary_writer.add_summary(summary, i)

            checkpoint_saver.save(sess, os.path.join( \
                self._checkpoint_dir, 'model'), global_step=iter_end)
            logger.info('Training ends after %d iterations.' % iter_end)

        summary_writer.close()
        self._val_summary_writer.close()

        logger.info('Training DCGAN ... Done')

    def test(self):
        """Evaluate the trained network using the whole test set.
        
        Note, the we sample random latent input for the generator.
        """
        if not self._is_build:
            raise CustomException('Network has not been build yet.')

        logger.info('Testing DCGAN ...')

        sess = self._get_inference_session()
        if sess is None:
            logger.error('Could not create session. Testing aborted.')

        test_ins = shared.data.get_test_inputs()
        test_outs = shared.data.get_test_outputs()
        test_latent_inputs = self.sample_latent(shared.data.num_test_samples)

        test_handle = sess.run(self._test_iter.string_handle())
        sess.run(self._test_iter.initializer,
             feed_dict={self._t_test_raw_in: test_ins,
                        self._t_test_raw_out: test_outs,
                        self._t_test_batch_size: shared.data.num_test_samples})

        ckpt_epoch = tf.train.global_step(sess, self._t_global_step)
        logger.info('The network has been trained for %d epochs.' % 
                    ckpt_epoch)

        # Note, that subclasses (such as a WassersteinGan), don't have a
        # meaningful accuracy.
        if self._t_accuracy is None:
            g_loss, d_loss = sess.run([self._g_loss, self._d_loss],
                feed_dict={self._g_inputs: test_latent_inputs,
                           self._t_handle: test_handle})
        else:
            acc, g_loss, d_loss = sess.run( \
                [self._t_accuracy, self._g_loss, self._d_loss],
                feed_dict={self._g_inputs: test_latent_inputs,
                           self._t_handle: test_handle})

            logger.info('Test Accuracy: %f' % acc)
        logger.info('Generator loss on test set: %f' % g_loss)
        logger.info('Discriminator loss on test set: %f' % d_loss)
        
        if self.allow_plots:
            num_plots = min(8, test_latent_inputs.shape[0])

            Z_in = test_latent_inputs[:num_plots, :]

            # We have to reinitialize to change the batch size (seems to be
            # a cleaner solution than processing the whole validation set).
            sess.run(self._test_iter.initializer,
                feed_dict={self._t_test_raw_in: test_ins[:num_plots, :],
                           self._t_test_raw_out: test_outs[:num_plots, :],
                           self._t_test_batch_size: num_plots})
            real_imgs, real_lbls, fake_imgs, fake_dis_outs, real_dis_outs = \
                sess.run([self._t_ds_inputs, self._t_ds_outputs,
                          self._g_outputs, self._d_outputs_fake,
                          self._d_outputs_real],
                feed_dict={self._g_inputs: Z_in,
                           self._t_handle: test_handle})

            dplt.plot_gan_images('Test Samples', real_imgs,
                                 fake_imgs, real_outputs=real_lbls,
                                 real_dis_outputs=real_dis_outs,
                                 fake_dis_outputs=fake_dis_outs,
                                 shuffle=True, interactive=True,
                                 figsize=(10, 12))

        logger.info('Testing DCGAN ... Done')

    def run(self, inputs, latent_inputs=None):
        """Run the network with the given inputs.

        Args:
            inputs: Samples that align with the dataset (2D numpy array).
                This would be send through the discriminator. Can be an empty
                array.
            latent_inputs: Inputs for the generator network. This is used to
                generate fake images.

        Returns:
            [real_dis_outs, fake_dis_outs, fake_imgs]
            real_dis_outs: The discriminator outputs for the real inputs
                (parameter 'inputs'). Is None, if 'inputs' is empty.
            fake_dis_outs: The discriminator outputs for the generated images.
            fake_imgs: The generator output, for the given latent input.
        """
        sess = self._get_inference_session()
        if sess is None:
            logger.error('Could not create session. Inference aborted.')

        fake_imgs = None
        real_dis_outs = None
        fake_dis_outs = None

        if np.size(inputs) != 0:
            [real_dis_outputs] = sess.run([self._d_outputs_real],
                feed_dict={self._d_inputs: inputs})

        if latent_inputs is not None:
            fake_imgs, fake_dis_outs = sess.run( \
                [self._g_outputs, self._d_outputs_fake],
                feed_dict={self._g_inputs: latent_inputs})

        return [real_dis_outs, fake_dis_outs, fake_imgs]

    def sample_latent(self, batch_size):
        """Samples random input from the latent (generator input) space.

        Args:
            batch_size: The batch size.

        Returns:
            Returns a 2D numpy array of shape (batch_size, latent_size).
        """
        return np.random.normal(loc=0.0, scale=1.0,
                                size=[batch_size, self.latent_size])

    def _init_validation(self, val_batch_size):
        """Initialize the validation process, that runs in parallel to the
        training.

        Args:
            val_batch_size: Size of the validation batch.
        """
        self._val_batch = shared.data.next_val_batch(val_batch_size)
        if self._val_batch is None:
            self._val_batch = shared.data.next_test_batch(val_batch_size)
        # Has to be the same, every time the validation batch is used.
        self._val_latent_input = self.sample_latent(val_batch_size)

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

        # Note, that subclasses (such as a WassersteinGan), don't have a
        # meaningful accuracy.
        if self._t_accuracy is None:
            g_loss, d_loss, summary = sess.run( \
                    [self._g_loss, self._d_loss, self._t_summaries],
                    feed_dict={self._g_inputs: self._val_latent_input,
                               self._t_handle: val_handle})
        else:
            acc, g_loss, d_loss, summary = sess.run( \
                    [self._t_accuracy, self._g_loss, self._d_loss,
                     self._t_summaries],
                    feed_dict={self._g_inputs: self._val_latent_input,
                               self._t_handle: val_handle})

            logger.info('Validation Accuracy: %f' % acc)
        logger.info('Generator loss on validation batch: %f' % g_loss)
        logger.info('Discriminator loss on validation batch: %f' % d_loss)
    
        self._val_summary_writer.add_summary(summary, epoch)
        self._val_summary_writer.flush()

        if self.allow_plots:
            num_plots = min(4, self._val_latent_input.shape[0])

            # We have to reinitialize to change the batch size (seems to be
            # a cleaner solution than processing the whole validation set).
            sess.run(self._val_iter.initializer,
                feed_dict={self._t_val_raw_in:
                               self._val_batch[0][:num_plots, :],
                           self._t_val_raw_out:
                               self._val_batch[1][:num_plots, :],
                           self._t_val_batch_size: num_plots})
            fake_imgs, fake_dis_outs = sess.run( \
                [self._g_outputs, self._d_outputs_fake],
                feed_dict={self._g_inputs:
                                self._val_latent_input[:num_plots, :],
                           self._t_handle: val_handle})

            dplt.plot_gan_images('Validation Samples', np.empty((0,0)),
                                 fake_imgs, fake_dis_outputs=fake_dis_outs,
                                 interactive=True)

        logger.info('Epoch %d: validating training process ... Done' % epoch)

    def _build_generator(self):
        """Builds the generator network, consisting of an initial fully-
        connected layer followed by several strided transpose convolutional
        layers.
        """
        biases_initializer = tf.zeros_initializer() if self.use_biases \
            else None

        # The initial fully-connected layer, that transforms the latent input
        #  vectors in such a way, that it can be feeded into the deconv layers.
        fc_out_shape = self._gen_fc_out_wof + [self._gen_filters[0]]
        with tf.variable_scope('hidden_0'):
            self._g_fc = layers.fully_connected(self._g_inputs,
                int(np.prod(fc_out_shape)), activation_fn=tf.nn.relu, 
                weights_initializer=self._initializer,
                biases_initializer=biases_initializer,
                use_bn=True, is_training=self._t_mode)
            self._g_fc = tf.reshape(self._g_fc, [-1] + fc_out_shape)

        last_layer = self._g_fc

        # Transpose convolutional layers
        self._g_hidden_deconv = []
        for i in range(1, self.num_gen_layers - 1):
            with tf.variable_scope('hidden_%d' % (i)):
                hidden = layers.conv2d_transpose(last_layer,
                    self._gen_filters[i], self._gen_kernels[i-1],
                    strides=self._gen_strides[i-1],
                    padding=self._gen_pads[i-1], activation=tf.nn.relu,
                    use_bias=self.use_biases,
                    kernel_initializer=self._initializer, use_bn=True,
                    is_training=self._t_mode)
                self._g_hidden_deconv.append(hidden)
                last_layer = hidden

        # The generator output. Note, that it uses no batchnorm and a
        # different activation function.
        i = self.num_gen_layers - 1
        with tf.variable_scope("output"):
            self._g_outputs = layers.conv2d_transpose(last_layer,
                self._gen_filters[i], self._gen_kernels[i-1],
                strides=self._gen_strides[i-1], padding=self._gen_pads[i-1],
                activation=tf.nn.tanh, use_bias=self.use_biases,
                kernel_initializer=self._initializer)

        tf.summary.image('fake', tf.reshape(self._g_outputs, \
                         [-1] + shared.data.in_shape))

    def _build_discriminator(self, inputs, reuse=False):
        """Build the discriminator network, which consists of consecutive
        strided convolutional layers (without pooling), followed by a fully-
        connected network with a single output.

        Args:
            inputs: The input tensor to the discriminator.
            reuse: Whether the created variables can be reused.

        Returns:
            [d_hidden, d_outputs]
            d_hidden is a list of hidden layer output tensors.
            d_outputs is the output tensor of the discriminator.
        """
        biases_initializer = tf.zeros_initializer() if self.use_biases \
            else None

        # Convolutional layers.
        last_layer = inputs
        d_hidden = []
        for i in range(self.num_dis_layers-1):
            with tf.variable_scope('hidden_%d' % (i), reuse=reuse):
                # Note, that we don't use batchnorm in the first layer of the
                # discriminator.
                if i == 0:
                    use_bn = False
                    is_training = None
                else:
                    use_bn = True
                    is_training = self._t_mode
                
                hidden = layers.conv2d(last_layer, self._dis_filters[i],
                    self._dis_kernels[i], strides=self._dis_strides[i],
                    padding=self._dis_pads[i], activation=tf.nn.leaky_relu,
                    kernel_initializer=self._initializer,
                    use_bias=self.use_biases, use_bn=use_bn,
                    is_training=is_training)
                d_hidden.append(hidden)
                last_layer = hidden

        # Final fully connected layer.
        with tf.variable_scope("output", reuse=reuse):
            curr_sample_size = int(np.prod(last_layer.shape[1:]))
            last_layer = tf.reshape(last_layer, [-1, curr_sample_size])
            d_outputs = layers.fully_connected(last_layer, 1,
                activation_fn=None, weights_initializer=self._initializer,
                biases_initializer=biases_initializer)

        return [d_hidden, d_outputs]

    def _compute_generator_layout(self):
        """This method will compute the parameters of the transpose
        convolutional layers of the generator network.

        The construction is described in the docstring of the setter method of
        the attribute "num_gen_layers".
        """
        # Only consider transpose convolutional layers.
        num_layers = self._num_glayers - 1

        # Output shape of the generator.
        gos = shared.data.in_shape
        assert(np.size(gos) == 3)

        pads = ['SAME'] * num_layers
        kernels = [[5,5]] * num_layers
        filters = [gos[-1]] * num_layers
        strides = [2] * num_layers

        # Width and height of the current layers output, starting from the end.
        cw = gos[0]
        ch = gos[1]

        for i in range(num_layers-1, -1, -1):
            # As default option, we use a similar number of filters as in the
            # paper.
            if i < num_layers-1:
                filters[i] = 128 * 2**(num_layers-1 - i - 1)
            
            if cw <= 4 or ch <= 4: # E.g., 4 -> 4
                # This is a condition we set.
                # If one of the sizes is too small, we just keep the layer
                # size, meaning we set a stride of 1 and 'SAME' padding.
                strides[i] = 1
                continue
            
            if cw % 2 == 0: # E.g., 8 -> 16
                # This layer should just double the size.
                # Hence, we keep a stride of 2. Since we are using 'SAME'
                # padding, the kernel size is arbitrary.
                pass
            else: # E.g., 3 -> 7
                # The input will have size cw // 2. To achieve this, we will
                # use valid padding with a stride of 2 and a kernel size of 3.
                # According to the formula, with p = 0, s = 2 and k = 3, we 
                # will get
                # out_size = 2 (in_size-1) + 3 - 2*0 = 2*in_size + 1

                kernels[i] = [3,3]
                pads[i] = 'VALID'

            # We do the same for ch.
            if ch % 2 == 1:
                kernels[i] = [3,3]
                pads[i] = 'VALID'

            # What if they both behave differently?
            if cw % 2 != ch % 2: # E.g., (4,3) -> (8,7)
                # In this case, we need to use valid padding, such that the
                # even number has a kernel size of 2 and the odd number has a
                # kernel size of 3.

                assert(pads[i] == 'VALID')

                if cw % 2 == 0:
                    kernels[i] = [2, 3]
                else:
                    kernels[i] = [3, 2]

            # Output sizes for previous layer.
            cw = cw // 2
            ch = ch // 2

        # Note, that the initial fully connected-layer also has a filter number.
        filters = [128 * 2**(num_layers-1)] + filters

        # The output shape (without filters) of the initial fully connected
        # layer.
        self._gen_fc_out_wof = [cw, ch]
        self._gen_strides = strides
        self._gen_pads = pads
        self._gen_kernels = kernels
        self._gen_filters = filters

    def _compute_discriminator_layout(self):
        """This method will compute the parameters of the convolutional layers
        of the discriminator network.
        
        The construction is described in the docstring of the setter method of
        the attribute "num_dis_layers".
        """

        # Only consider convolutional layers.
        num_layers = self._num_dlayers - 1

        # Input shape of the discriminator.
        dis = shared.data.in_shape
        assert(np.size(dis) == 3)

        pads = ['SAME'] * num_layers
        kernels = [[5,5]] * num_layers
        filters = [64] * num_layers
        strides = [2] * num_layers

        # Width and height of the current layers output, starting from the end.
        cw = dis[0]
        ch = dis[1]

        for i in range(num_layers):
            filters[i] = 64 * 2**i

            if cw <= 4 or ch <= 4:
                # This is a condition we set.
                # If one of the sizes is too small, we just keep the layer
                # size, meaning we set a stride of 1.
                strides[i] = 1
                continue

            cw = int(np.ceil(cw / 2))
            ch = int(np.ceil(ch / 2))

        self._dis_strides = strides
        self._dis_pads = pads
        self._dis_kernels = kernels
        self._dis_filters = filters

    def _compute_loss(self):
        """Compute the loss of the generator and discriminator.
        
        This method will set the two attributes _g_loss and _d_loss.
        
        Note, in this implementation, we also compute the accuracy of the
        discriminator.
        """
        # For the discriminator, we minimize
        #   -log(D(img)) - log(1-D(G(z))) (normal cross entropy).
        # Note, we apply one-sided label smoothing.
        self._d_loss = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
                labels=tf.zeros_like(self._d_outputs_fake),
                logits=self._d_outputs_fake)) + \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
                labels=.9 * tf.ones_like(self._d_outputs_real),
                logits=self._d_outputs_real))
        
        # Here, we minimize -log(D(G(z))) 
        self._g_loss = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
                labels=tf.ones_like(self._d_outputs_fake),
                logits=self._d_outputs_fake))

        correct_prediction = tf.concat([self._d_outputs_real > 0.5,
                                        self._d_outputs_fake <= 0.5],
                                       axis=0)
        self._t_accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                  tf.float32))

    def _get_optimizers(self, op_params):
        """Instantiate the optimizers used by this network.

        Args:
            op_params: A dictionary of parameters for the optimizers.

        Returns:
            A list of two optimizer objects. One for the generator and one for
            the discriminator.
        """
        gen_optimizer = tf.train.AdamOptimizer( \
            learning_rate=op_params['learning_rate'],
            beta1=op_params['beta1'], beta2=op_params['beta2'])

        dis_optimizer = tf.train.AdamOptimizer( \
            learning_rate=op_params['learning_rate'],
            beta1=op_params['beta1'], beta2=op_params['beta2'])

        return [gen_optimizer, dis_optimizer]

if __name__ == '__main__':
    pass
