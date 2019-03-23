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
@title           :configuration.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/09/2018
@version         :1.0
@python_version  :3.6.6

All the global configurations are specified in this file.
"""

import logging
from datetime import datetime

########################################
### Network
########################################
# The framework is designed to provide a simple way for users to build,
# train and run ANN networks with different datasets.
# Instead of providing a colossal configuration file, the user writes simple
# scripts that load the dataset he is interested in and create the desired
# networks. Examples of such scripts can be found in the folder "examples".
# Importantly, these scripts must contain a run(**kwargs) method that can be
# executed.

# Please, provide the path to such a script.
#run_script = 'examples.mnist_autoencoder'
#run_script = 'examples.mnist_dcgan'
#run_script = 'examples.mnist_wgan'
#run_script = 'examples.mnist_classifier'
#run_script = 'examples.cifar10_dcgan'
#run_script = 'examples.cifar10_wgan'
#run_script = 'examples.celeba_dcgan'
#run_script = 'examples.celeba_wgan'
run_script = 'examples.mi_toy_sample'

########################################
### Dataset
########################################

# Where to read from resp. store the dataset?
dataset_path = '../data'

########################################
### Outputs
########################################

# Where should all the outputs of a program run should be stored?
output_folder = '../out'
#output_folder = '../out' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# If existing output folders should be deleted. If not activated, it will try
# to merge results, which may be confusing. Additionally, everything that has
# to be deleted during the programm run, will be deleted if this is false.
# However, if you have several testing sessions then you don't wanna delete all
# outputs, as this might delete your trained network.
clear_outputs = False

# Ask the user before deleting an existing output folder. Only relevant if
# "clear_outputs" is enabled.
ask_before_del = True

########################################
### Logging
########################################
# These options are only concerned about the console logging.

# Set log level (Choose one of the following: DEBUG, INFO, WARNING, ERROR,
# CRITICAL).
file_loglevel = logging.DEBUG
console_loglevel = logging.DEBUG

########################################
### Miscellaneous
########################################

# A random seed will enforce reproducible behavior.
# If set to None, no random seed will be set.
random_seed = 42

if __name__ == '__main__':
    pass


