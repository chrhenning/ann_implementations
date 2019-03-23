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
@title           :examples/mnist_classifier.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/23/2018
@version         :1.0
@python_version  :3.6.6

This script shows how to use the class CNNClassifier together with the MNIST
dataset.
"""
import configuration as config
import misc.shared_vars as shared
from data.mnist_data import MNISTData
from networks.classifier.cnn_classifier import CNNClassifier

import logging
logger = logging.getLogger(shared.logging_name)

ALLOW_PLOTS = True
CONTINUE_TRAINING = False
INFERENCE_CPU_ONLY = False
USE_BATCHNORM = False

def setup_network(allow_plots, continue_training, inference_cpu_only,
                  use_batchnorm, mode='inference'):
    """Setup the MNIST network. Note, that this method will already build the
    network.

    Args:
        allow_plots: This option should only be enabled if a graphical backend
                     is available.
        continue_training: Continue training from existing checkpoint.
        inference_cpu_only: Do not run inference (testing session) on the GPU.
        use_batchnorm: Allow the use of batchnorm in the network.
        mode: The network mode.

    Returns:
        The instantiated network.
    """
    net = CNNClassifier(mode=mode)
    net.log_device_placement(True)
    net.dynamic_gpu_mem = True
    if mode == 'train':
        net.continue_training = continue_training
    else:
        net.cpu_only = inference_cpu_only
    net.allow_plots = allow_plots
    net.use_batchnorm = use_batchnorm

    net.build()

    return net

def run(**kwargs):
    allow_plots = ALLOW_PLOTS
    continue_training = CONTINUE_TRAINING
    inference_cpu_only = INFERENCE_CPU_ONLY
    use_batchnorm = USE_BATCHNORM
    
    for k in kwargs.keys():
        if k == 'allow_plots':
            allow_plots = kwargs[k]
        elif k == 'continue_training':
            continue_training = kwargs[k]
        elif k == 'inference_cpu_only':
            inference_cpu_only = kwargs[k]
        elif k == 'use_batchnorm':
            use_batchnorm = kwargs[k]
        else:
            logger.warn('Keyword \'%s\' is unknown.' % k)
    
    logger.info('### Loading dataset ...')

    data = MNISTData(config.dataset_path, use_one_hot=True)

    # Important! Let the network know, which dataset to use.
    shared.data = data

    logger.info('### Loading dataset ... Done')

    logger.info('### Build, train and test network ...')

    train_net = setup_network(allow_plots, continue_training,
                              inference_cpu_only, use_batchnorm, mode='train')

    train_net.train(num_iter=10001)

    test_net = setup_network(allow_plots, continue_training,
                             inference_cpu_only, use_batchnorm,
                             mode='inference')
    test_net.test()


    if allow_plots:
        # Example Test Samples
        sample_batch = data.next_test_batch(8)
        predictions = test_net.run(sample_batch[0])
        shared.data.plot_samples('Example MNIST Predictions', sample_batch[0],
                                     outputs=sample_batch[1],
                                     predictions=predictions,
                                     interactive=True)

    logger.info('### Build, train and test network ... Done')

if __name__ == '__main__':
    pass