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
@title           :examples/mnist_dcgan.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/16/2018
@version         :1.0
@python_version  :3.6.6

This script shows how to use the class DCGAN together with the MNIST
dataset.
"""
import numpy as np

import configuration as config
import misc.shared_vars as shared
from data.mnist_data import MNISTData
from networks.gan.dcgan import DCGAN
import misc.visualizations.plotting_data as dplt

import logging
logger = logging.getLogger(shared.logging_name)

ALLOW_PLOTS = True
CONTINUE_TRAINING = False

def setup_network(allow_plots, continue_training, mode='inference'):
    """Setup the MNIST network. Note, that this method will already build the
    network.

    This method will enforce a certain kernel size to all layers (if possible).

    Args:
        allow_plots: This option should only be enabled if a graphical backend
                     is available.
        continue_training: Continue training from existing checkpoint.
        mode: The network mode.

    Returns:
        The instantiated network.
    """
    net = DCGAN(mode=mode)
    if mode == 'train':
        net.continue_training = continue_training
    net.allow_plots = allow_plots

    for i in range(1, net.num_gen_layers):
        if net.get_padding_gen(i).lower() == 'same':
            net.set_kernel_size_gen([3,3], i)
    for i in range(net.num_dis_layers-1):
        if net.get_padding_dis(i).lower() == 'same':
            net.set_kernel_size_dis([3,3], i)

    net.build()

    return net

def run(**kwargs):
    allow_plots = ALLOW_PLOTS
    continue_training = CONTINUE_TRAINING
    
    for k in kwargs.keys():
        if k == 'allow_plots':
            allow_plots = kwargs[k]
        elif k == 'continue_training':
            continue_training = kwargs[k]
        else:
            logger.warn('Keyword \'%s\' is unknown.' % k)

    logger.info('### Loading dataset ...')

    data = MNISTData(config.dataset_path)

    # Important! Let the network know, which dataset to use.
    shared.data = data

    logger.info('### Loading dataset ... Done')

    logger.info('### Build, train and test network ...')

    train_net = setup_network(allow_plots, continue_training, mode='train')
    train_net.train(num_iter=10001)
    
    test_net = setup_network(allow_plots, continue_training, mode='inference')
    test_net.test()

    if allow_plots:
        # Generate some fake images.
        latent_inputs = test_net.sample_latent(8)
        _, fake_dis_outs, fake_imgs = test_net.run(np.empty((0,0)),
                                                   latent_inputs=latent_inputs)
        dplt.plot_gan_images('Generator Samples', np.empty((0,0)), fake_imgs,
                             fake_dis_outputs=fake_dis_outs, shuffle=True, 
                             interactive=True)

    logger.info('### Build, train and test network ... Done')

if __name__ == '__main__':
    pass
