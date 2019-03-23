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
@title           :main.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/09/2018
@version         :1.0
@python_version  :3.6.6

A wrapper script to setup an environment for running example scripts, that use
the networks and datasets implemented in this package.
"""

import time
import random
import numpy as np
import tensorflow as tf
import shutil
import os
import argparse
import json
import sys

import configuration as config
from misc import logger_config
import misc.shared_vars as shared
import importlib

if __name__ == '__main__':
    start_time = time.time()

    # Parse arguments.
    parser = argparse.ArgumentParser(
        description="A wrapper script to easily train and test your " +
                    "network configurations with this package.")
    parser.add_argument('-r', '--run', type=str, default=config.run_script,
                        metavar='MODULE_PATH',
                        help="The path of a python module, containing a " +
                             "'run' function. This function has to load the " +
                             "dataset and work with a network.")
    parser.add_argument('-i', '--data_dir', type=str,
                        default=config.dataset_path,
                        help="Where are the datasets stored? If a dataset " +
                             "is not existing, it may be stored into this" +
                             "folder (if automatic download supported).")
    parser.add_argument('-o', '--out_dir', type=str,
                        default=config.output_folder,
                        help="Where should the outputs of the program be " +
                             "stored?")
    parser.add_argument('-d', '--del_old', action='store_true' \
                            if not config.clear_outputs else 'store_false',
                        help="Delete output folder if existing.")
    parser.add_argument('--random_seed', type=int, default=config.random_seed,
                        help="Random seed.")
    parser.add_argument('-k', '--kwargs', type=str, default="{}",
                        help="If the chosen run script accepts keyword " +
                             "arguments, they can be passed via this " +
                             "argument as a JSON dictionary (encoded as " +
                             "string).")
    args = parser.parse_args()

    # Overwrite default configs.
    config.run_script = args.run
    config.dataset_path = args.data_dir
    config.output_folder = args.out_dir
    config.clear_outputs = args.del_old
    config.random_seed = args.random_seed
    kwargs = json.loads(args.kwargs)

    # Import the module whose run function should be executed.
    run_module = importlib.import_module(config.run_script)

    # Check output folder.
    old_outs_removed = False
    
    if config.clear_outputs:
        if os.path.exists(config.output_folder):
            if config.ask_before_del:
                response = input('The output folder %s already exists. ' % + \
                                 (config.output_folder) + \
                                 'Do you want us to delete it? [y/n]')
                if response != 'y':
                    raise Exception('Could not delete output folder!')
    
            shutil.rmtree(config.output_folder)
            old_outs_removed = True

    if not os.path.exists(config.output_folder):
        os.mkdir(config.output_folder)

    logger = logger_config.config_logger(shared.logging_name, 
        os.path.join(config.output_folder, 'logfile.txt'), 
        config.file_loglevel, config.console_loglevel)
    logger.info('### Another ANN Wrapper ###')

    if old_outs_removed:
        logger.warn('Existing output directory had to be removed!')

    # Make all random processes predictable.
    if config.random_seed is not None:
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        tf.set_random_seed(config.random_seed)

    # Run an a priori defined example:
    logger.info('### Running file: %s' % config.run_script)
    if not isinstance(kwargs, dict):
        logger.error("The provided keyword arguments are not a valid " +
                     "dictionary.")
        sys.exit(1)

    if len(kwargs.keys()) > 0:
        logger.info('### The following keyword arguments for the run script ' +
                    'have been provided: %s' % kwargs)

    if not 'run' in dir(run_module):
        logger.error('There is no run method in file: %s' % config.run_script)
        sys.exit(2)

    run_module.run(**kwargs)

    end_time = time.time()
    logger.info('### Overall Runtime: %f sec' % (end_time - start_time))


