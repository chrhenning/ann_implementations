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
@title           :networks/gan/wasserstein_gan.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/17/2018
@version         :1.0
@python_version  :3.6.6

This is an implementation of a WassersteinGAN, as described in algorithm 1 in
the paper:
    https://arxiv.org/abs/1701.07875
"""
import tensorflow as tf
import os

from networks.gan.dcgan import DCGAN
import misc.shared_vars as shared
from misc.custom_exceptions.custom_exception import CustomException

import logging
logger = logging.getLogger(shared.logging_name)

class WassersteinGAN(DCGAN):
    """An implementation of a Wasserstein GAN that uses the architecture of a
    DCGAN

    Attributes (additional to base class):
    """    
    def __init__(self, mode='train'):
        """
        Args:
            mode: The value of the attribute "mode".
        """
        super().__init__(mode)

    def train(self, num_iter=10000, batch_size=64, learning_rate=0.00005, \
              n_critic=5, clip_val=0.01, val_interval=1000, val_bs=1000):
        """Train the network.

        The network is trained via the RMSProp optimizer.

        Note, if no validation set is available, the test set will be used.

        Args:
            num_iter: The number of training iterations.
            batch_size: The training batch size.
            learning_rate: See docs of "tf.train.RMSPropOptimizer".
            n_critic: The number of update steps for the critic per iteration.
                Note, that the critic should be trained to convergence before
                updating the discriminator.Note, that this number might be
                sporadically changed by the code in the function.
            clip_val: This implementation applies weight clipping to ensure
                the Lipschitz constraint of the critic (as proposed in the
                original paper).
            val_interval: How often the training status should be validated.
            val_bs: The batch size of the validation set to use.
        """
        if not self._is_build:
            raise CustomException('Network has not been build yet.')
            
        logger.info('Training WassersteinGAN ...')

        with self._graph.as_default() as g:
            summary_writer = tf.summary.FileWriter( \
                self._get_summary_dir('train_summary',
                                      rm_existing=not self.continue_training),
                                      g)

            self._init_validation(val_bs)
            
            # We need an additional operation, that clips the weights of the
            # critic after each training step.
            critic_w_clipping_op = \
                [w.assign( \
                        tf.clip_by_value(w, -clip_val, clip_val)) \
                 for w in self._d_vars]

            op_params = {'learning_rate': learning_rate}
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

                # Train critic until convergence:
                # This trick is borrowed from the original code:
                #   https://git.io/fAfst
                # Allegedly, this should only ensure that the critic becomes
                # strong very quickly, such that we have nice critic loss plots
                # (which correspond to the EM distance).
                n_critic_used = n_critic
                if i < 25 or i % 500 == 0:
                    n_critic_used = 100

                for t in range(n_critic_used):
                    feed_dict={self._g_inputs: self.sample_latent(batch_size),
                               self._t_handle: train_handle,
                               self._t_mode: True}

                    if t == 0:
                        _, summary = sess.run(
                            [dis_train_step, self._t_summaries],
                            feed_dict=feed_dict)
                        summary_writer.add_summary(summary, i)
                    else:
                        sess.run([dis_train_step], feed_dict=feed_dict)

                    # Clip weights.
                    sess.run(critic_w_clipping_op)

                # Train generator.
                # FIXME: Why does tf want me to feed real images into this run?
                sess.run([gen_train_step],
                    feed_dict={self._g_inputs: self.sample_latent(batch_size),
                               self._t_handle: train_handle,
                               self._t_mode: True})

            checkpoint_saver.save(sess, os.path.join( \
                self._checkpoint_dir, 'model'), global_step=iter_end)
            logger.info('Training ends after %d iterations.' % iter_end)

        summary_writer.close()
        self._val_summary_writer.close()

        logger.info('Training WassersteinGAN ... Done')

    def _compute_loss(self):
        """Compute the loss of the generator and discriminator.

        This method will set the two attributes _g_loss and _d_loss.
        """
        d_outputs_fake_mean = tf.reduce_mean(self._d_outputs_fake)

        # For the discriminator, we minimize
        self._d_loss = -(tf.reduce_mean(self._d_outputs_real) - 
                         d_outputs_fake_mean)

        # For the generator, we minimize
        self._g_loss = -d_outputs_fake_mean

    def _get_optimizers(self, op_params):
        """Instantiate the optimizers used by this network.

        Args:
            op_params: A dictionary of parameters for the optimizers.

        Returns:
            A list of two optimizer objects. One for the generator and one for
            the discriminator.
        """
        # The original paper proposes to use RMSProp without momentum.
        gen_optimizer = tf.train.RMSPropOptimizer( \
            learning_rate=op_params['learning_rate'])
        dis_optimizer = tf.train.RMSPropOptimizer( \
            learning_rate=op_params['learning_rate'])

        return [gen_optimizer, dis_optimizer]

if __name__ == '__main__':
    pass


