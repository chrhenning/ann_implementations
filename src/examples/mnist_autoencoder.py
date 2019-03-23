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
@title           :examples/mnist_autoencoder.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/14/2018
@version         :1.0
@python_version  :3.6.6

This script shows how to use the class SimpleAE together with the MNIST
dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

import configuration as config
import misc.shared_vars as shared
from data.mnist_data import MNISTData
from networks.autoencoder.simple_ae import SimpleAE

import logging
logger = logging.getLogger(shared.logging_name)

ALLOW_PLOTS=True

def run(**kwargs):
    allow_plots = ALLOW_PLOTS
    
    for k in kwargs.keys():
        if k == 'allow_plots':
            allow_plots = kwargs[k]
        else:
            logger.warn('Keyword \'%s\' is unknown.' % k)

    logger.info('### Loading dataset ...')

    data = MNISTData(config.dataset_path)

    # Important! Let the network know, which dataset to use.
    shared.data = data

    logger.info('### Loading dataset ... Done')


    logger.info('### Build, train and test network ...')

    # Train the network
    train_net = SimpleAE(mode='train')
    train_net.allow_plots = allow_plots
    train_net.build()
    train_net.train()

    # Test the network
    test_net = SimpleAE(mode='inference')
    test_net.allow_plots = allow_plots
    test_net.build()
    test_net.test()

    if allow_plots:
        # Feed a random test sample through the network and display the output
        # for the user.
        sample = data.next_test_batch(1)
        net_out = test_net.run(sample[0])
    
        fig = plt.figure()
        plt.ion()
        plt.suptitle('Sample Image')
        ax = fig.add_subplot(1,2,1)
        ax.set_axis_off()
        ax.imshow(np.squeeze(sample[0].reshape(data.in_shape)),
                  vmin=-1.0, vmax=1.0)
        ax.set_title('Input')
        ax = fig.add_subplot(1,2,2)
        ax.set_axis_off()
        ax.imshow(np.squeeze(net_out.reshape(data.in_shape)),
                  vmin=-1.0, vmax=1.0)
        ax.set_title('Output')
        plt.show()

    logger.info('### Build, train and test network ... Done')

if __name__ == '__main__':
    pass

