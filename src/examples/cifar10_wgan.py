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
@title           :examples/cifar10_wgan.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/17/2018
@version         :1.0
@python_version  :3.6.6

This script shows how to use the class WassersteinGAN together with the
CIFAR-10 dataset.
"""
import numpy as np

import configuration as config
import misc.shared_vars as shared
from data.cifar10_data import CIFAR10Data
from networks.gan.wasserstein_gan import WassersteinGAN
import misc.visualizations.plotting_data as dplt

import logging
logger = logging.getLogger(shared.logging_name)

ALLOW_PLOTS=True
USE_BIASES = True

def run(**kwargs):
    allow_plots = ALLOW_PLOTS
    use_biases = USE_BIASES
    
    for k in kwargs.keys():
        if k == 'allow_plots':
            allow_plots = kwargs[k]
        elif k == 'use_biases':
            use_biases = kwargs[k]
        else:
            logger.warn('Keyword \'%s\' is unknown.' % k)

    logger.info('### Loading dataset ...')

    data = CIFAR10Data(config.dataset_path)

    # Important! Let the network know, which dataset to use.
    shared.data = data

    logger.info('### Loading dataset ... Done')

    logger.info('### Build, train and test network ...')

    train_net = WassersteinGAN(mode='train')
    train_net.continue_training = False
    train_net.allow_plots = allow_plots
    train_net.use_biases = use_biases
    train_net.build()
    train_net.train(num_iter=100001)
    
    test_net = WassersteinGAN(mode='inference')
    test_net.allow_plots = allow_plots
    test_net.use_biases = use_biases
    test_net.build()
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
