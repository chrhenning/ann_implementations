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
@title           :networks/network_base_tf.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/10/2018
@version         :1.0
@python_version  :3.6.6

A wrapper for class NetworkBase, that provides common tensorflow
functionalities to its subclasses.
"""

import tensorflow as tf
import os
import shutil
import time
import numpy as np

from networks.network_base import NetworkBase
import configuration as config
import misc.shared_vars as shared
from misc.custom_exceptions.custom_exception import CustomException
from data.large_img_dataset import LargeImgDataset

import logging
logger = logging.getLogger(shared.logging_name)

class NetworkBaseTF(NetworkBase):
    """This class provides common tensorflow functionalities to its subclasses,
    such as the handling of checkpoints.

    Note, that this class is abstract and should not be instantiated.

    Attributes (additional to base class):
        cpu_only (default: False): Whether the sessions created with this
            network will run in CPU-only mode (i.e., not utilizing GPUs).
        val_cpu_only (default: False): Similar to the attribute cpu_only, but
            only the validation (running in parallel to the training) will be
            affected by this option.
            Note, this attribute only takes affect in classes, that have
            separate sessions for training and validation (such as the class
            SimpleAE).
        dynamic_gpu_mem (default: False): Allows a dynamic (as needed) GPU
            memory utilization (see Tensorflow "gpu_options.allow_growth").
        gpu_mem_fraction (default: None): The total fraction of GPU memory,
            that can be used by this process (see Tensorflow 
            "gpu_options.per_process_gpu_memory_fraction"). If set to None,
            the whole GPU memory may be used.
    """
    
    def __init__(self, mode='train'):
        """
        Args:
            mode: The value of the attribute "mode".
        """
        super().__init__(mode)

        self._checkpoint_dir =  os.path.join(config.output_folder,
                                             'checkpoints')

        self._scope_name = 'graph_scope'

        # If the network is in inference mode, it might be sensible to restore
        # the weights once and then keep the session in memory.
        self._inference_session = None

        self._cpu_only = False
        self._val_cpu_only = False
        self._log_device_placement = False
        self._dynamic_gpu_mem = False
        self._gpu_mem_fraction = None

        self._weight_init_path = None
        self._weight_init_vars = None

        # The following attributed are set when the method _make_dataset is
        # called for the first time.
        self._t_handle = None
        #self._data_iterator = None
        self._t_ds_inputs = None
        self._t_ds_outputs = None

        # The following attributes are set by the method _buil_datasets.
        self._t_train_raw_in = None
        self._t_train_raw_out = None
        self._train_iter = None
        self._t_train_batch_size = None
        self._t_val_raw_in = None
        self._t_val_raw_out = None
        self._val_iter = None
        self._t_val_batch_size = None
        self._t_test_raw_in = None
        self._t_test_raw_out = None
        self._test_iter = None
        self._t_test_batch_size = None

    @property
    def cpu_only(self):
        """Getter for the attribute cpu_only."""
        return self._cpu_only

    @cpu_only.setter
    def cpu_only(self, value):
        """Setter for the attribute cpu_only.

        Note, this will only take affect for sessions that have not been
        created yet.
        """
        self._cpu_only = value

    @property
    def val_cpu_only(self):
        """Getter for the attribute val_cpu_only."""
        return self._val_cpu_only

    @val_cpu_only.setter
    def val_cpu_only(self, value):
        """Setter for the attribute val_cpu_only."""
        self._val_cpu_only = value

    @property
    def dynamic_gpu_mem(self):
        """Getter for the attribute dynamic_gpu_mem."""
        return self._dynamic_gpu_mem

    @dynamic_gpu_mem.setter
    def dynamic_gpu_mem(self, value):
        """Setter for the attribute dynamic_gpu_mem.

        Note, this will only take affect for sessions that have not been
        created yet.
        """
        self._dynamic_gpu_mem = value

    @property
    def gpu_mem_fraction(self):
        """Getter for the attribute gpu_mem_fraction."""
        return self._gpu_mem_fraction

    @gpu_mem_fraction.setter
    def gpu_mem_fraction(self, value):
        """Setter for the attribute gpu_mem_fraction.

        Note, this will only take affect for sessions that have not been
        created yet.
        """
        self._gpu_mem_fraction = value

    def init_weights_from(self, ckpt_path, var_names=None):
        """Before training, initialize the weights of all network variables
        from a checkpoint.

        Note, this method only has an affect, if the attribute
        'continue_training' is not True.

        Note, this function will ensure, that the global_step is not restored
        (as this may mess with the training epochs).

        Args:
            ckpt_path: The path to the checkpoint file.
            var_names (optional): A list of variable names. If provided, only
                these variables are going to be loaded from the checkpoint.

        Examples:
            >>> net.init_weights_from('../out/checkpoints/model-1000',
            ...     ['scope_name/hidden_0/fully_connected/weights:0',
            ...      'scope_name/hidden_0/fully_connected/biases:0']
            ... )
        """
        self._weight_init_path = ckpt_path
        self._weight_init_vars = var_names

    def set_checkpoint_dir(self, ckpt_dir):
        """This method can be used to change the default checkpoint dir,
        which is located within the globally configured output directory and
        the same for all networks.

        Args:
            ckpt_dir: The new checkpoint directory, in which checkpoints will
                be written to and read from.
        """
        self._checkpoint_dir = ckpt_dir

    def log_device_placement(self, enable):
        """Whether device placements should be logged when a new tf.Session is
        created.

        See class tf.ConfigProto for details.

        Args:
            enable: A boolean, indicating whether device placement logging is
                enabled or not.
        """
        self._log_device_placement = enable

    def set_scope_name(self, sname):
        """Set the name of the scope defined by tf.name_scope. This scope name
        can later be used to distinguish different network instances, for
        example in Tensorboard.

        Note, this method has to be called before the network was built!
        
        Note, this affects only the major name scope within a graph. Different
        class instances (and therefore different graphs) are still managed by
        different Tensorbiard instances.

        Args:
            sname (default: 'graph_scope'): A string, defining the named scope
                of the graph.
        """
        # Note, we assume here, that all sublcasses correctly set the is_build
        # attribute.
        if self._is_build:
            raise CustomException('The name scope has to be set before the ' +
                                  'network was built.')

        self._scope_name = sname

    def _get_config_proto(self, cpu_only=None):
        """Get the configuration that should be used for tf.Session objects
        instantiated within this class.

        Args:
            cpu_only (optional): If provided, the class attribute 'cpu_only'
                will be ignored. Instead, this parameter will be used to
                determine, whether the new session should run on cpu only.

        Returns:
            A tf.ConfigProto object.
        """
        no_gpu = self.cpu_only
        if cpu_only is not None:
            no_gpu = cpu_only

        config = tf.ConfigProto(
            log_device_placement=self._log_device_placement
        )

        if no_gpu:
            config = tf.ConfigProto(
                device_count={'GPU': 0},
                log_device_placement=self._log_device_placement
            )

        config.gpu_options.allow_growth = self._dynamic_gpu_mem
        if self._gpu_mem_fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = \
                self._gpu_mem_fraction

        return config

    def _get_summary_dir(self, ident, rm_existing=True):
        """Get a directory, where we can save Tensorboard summaries. This
        method will delete already existing summary paths.
    
        Args:
            ident: The parent folder name of the summaries.
            rm_existing (default: True): If set to false, existing summary
                directories would not be deleted.
    
        Returns:
            The directory path as string.
        """
        summary_dir = os.path.join(config.output_folder, ident)
    
        if rm_existing and os.path.exists(summary_dir):
            logger.warn('Existing summary in folder %s will be deleted.' % \
                        summary_dir)
            shutil.rmtree(summary_dir)

        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)
        logger.info('Current summaries can be viewed via Tensorboard ' \
                    + 'from: %s' % summary_dir)
    
        return summary_dir
    
    def _remove_old_checkpoints(self):
        """Remove old checkpoints, if existing."""
        if os.path.exists(self._checkpoint_dir):
            logger.warn('Existing checkpoints in folder %s will be ' % \
                        self._checkpoint_dir + 'deleted.')
            shutil.rmtree(self._checkpoint_dir)
        os.mkdir(self._checkpoint_dir)
    
    def _latest_checkpoint_path(self):
        """Get the path of the latest checkpoint file.
        
        Returns:
            None, if no checkpoint file exists. The checkpoint path otherwise.
        """
        ckpt_path = tf.train.latest_checkpoint(self._checkpoint_dir)
        if not ckpt_path:
            # We try once more, before we give up.
            time.sleep(1)
            ckpt_path = tf.train.latest_checkpoint(self._checkpoint_dir)

        if not ckpt_path:
            logger.error('Could not find any checkpoint files!')
            return None
        # TODO: check that "epoch" is part of filename or choose different
        # one.

        return ckpt_path

    def _get_inference_session(self):
        """Create a session, that restores weights from the latest checkpoint.
        This session is left open for the lifetime of this object (it is closed
        in the destructor of the class).
        """
        assert(self._is_build)
        if self._inference_session is None:
            ckpt_path = tf.train.latest_checkpoint(self._checkpoint_dir)
            if not ckpt_path:
                logger.error('Could not restore network weights. No ' + \
                         'checkoint found in %s.' % self._checkpoint_dir)
                return None

            with self._graph.as_default():
                saver = tf.train.Saver()

            self._inference_session = tf.Session(graph=self._graph, \
                config=self._get_config_proto())
            saver.restore(self._inference_session, ckpt_path)

        return self._inference_session

    def _init_training(self, sess, num_iter, init_op, global_step, ckpt_saver):
        """This method can be used to initialize the training, i.e., to either
        resume a prior training process (depending on the attribute
        'continue_training') or start a new one (and thus initialize all
        weights).

        Note, if 'continue_training' enabled, but no checkpoint is found, the
        training will start from scratch.

        Args:
            sess: The tf session used for training.
            num_iter: The number of training iterations, that should be
                performed in this run.
            init_op: A tf initialization op, that can be run to initialize all
                variables (in case, that they are not restored).
            global_step: A tensor, describing the networks global step.
            ckpt_saver: A tf.train.saver object.

        Returns:
            A tuple of integers:
                iter_start: The first global step of the now starting training
                    step.
                iter_end: The training run until it reaches this epoch number.
        """
        iter_start = 0
        iter_end = num_iter

        # Restore training, if requested.
        if self.continue_training:
            ckpt_path = self._latest_checkpoint_path()
            if not ckpt_path:
                logger.error('Starting training from scratch.')
                self._remove_old_checkpoints()
                sess.run(init_op)
            else:
                ckpt_saver.restore(sess, ckpt_path)
                ckpt_epoch = tf.train.global_step(sess, global_step)
                logger.info('Resuming training from iteration %d.' % \
                            ckpt_epoch)
                iter_start = ckpt_epoch
                iter_end = iter_start + num_iter
        else:
            # It won't take too long, just initilize all variables and
            # then overwrite the once that should be restored.
            sess.run(init_op)

            # Initialize variables from checkpoint.
            if self._weight_init_path is not None:
                if self._weight_init_vars is not None:
                    restore_vars = [v for v in tf.global_variables() \
                                    if v.name in self._weight_init_vars]
                else:
                    restore_vars = [v for v in tf.global_variables()]

                restore_vars = [v for v in restore_vars if v != global_step]

                logger.info('Variables [%s] will be restored from %s.' %  \
                            (', '.join(str(v.name) for v in restore_vars),
                             self._weight_init_path))

                init_saver = tf.train.Saver(restore_vars)
                init_saver.restore(sess, self._weight_init_path)

            # Any old checkpoints are going to be removed and replaced by 
            self._remove_old_checkpoints()

        return [iter_start, iter_end]

    def _make_dataset(self, t_data_in, t_data_out,
                      mode='inference', pp_map=None, shuffle=None):
        """Create a dataset according to the tf.data.Dataset interface.

        For instance, one should create a dataset for training, validation and
        test sets. This method allows one to circumvent the feed_dict, usually
        used in Tensorflow. Additionally, it provides an easy way to preprocess
        data and read the actual data into memory on demand. To accomplish
        this, the input and output map function defined in the Dataset class
        are used.

        This method will set several class attributes when called for the first
        time.
         - self._t_handle: A string placeholder, that can be used to easily
                           switch between different datasets.
         - self._t_ds_inputs: A tensor representing the actual input batch
                              (so this should be the input tensor to the
                              network).
           self._t_ds_outputs: A tensor representing the actual output batch.

        If it should still be allowed to feed arbitrary data to the network
        (without the need of building an extra dataset), one could use the
        tf.placeholder_with_default option (where the default value would
        come from the dataset).

        The data from which the dataset should be constructed (e.g., the
        complete training set) should not be fed in directly, as it would
        become part of the graph. Instead, we use placeholders, that can be
        used to fed the raw data into the graph when initializing the dataset
        (its iterator). Note, mapping function will be used, as soon as a batch
        is actually needed. So, the complete dataset might contain only
        filenames whereas the mapping function will load the data into memory
        only if needed.

        Args:
            t_data_in: A placeholder, that is used to fed the input data into
                       the dataset.
            t_data_out: A placeholder, that is used to fed the output data into
                        the dataset.
            mode (default: 'inference'): This argument will be passed to the
                                         dataset specific mapping function.
            pp_map (optional): A function handle, that can be used to
                               preprocess the data samples. The function will
                               take the input and output sample as argument and
                               return a modified input and output tensor.
            shuffle (optional): If the samples of the dataset should be
                                presented in random order, then a certain
                                amount of them has to be preloaded. This option
                                is used to set this number (if not provided,
                                order will be deterministic).

        Returns:
            A tuple (iterator, t_batch_size). The iterator object should be
            used to create a string handle, that tells Tensorflow which dataset
            to use. The batch size tensor has to be used to set a custom batch
            size when reinitializing the dataset.
        """
        # Mapping functions, that prepare the data (e.g., reading it from
        # memory).
        input_map = shared.data.tf_input_map(mode)
        output_map = shared.data.tf_output_map(mode)

        # Note, we cannot feed the data directly into the dataset, as it would
        # become part of the graph (the graph is stored when doing checkpoints
        # and would be to huge for large datasets). Instead, we feed in a
        # placeholder.
        # See https://git.io/fAQG3.
        dataset_in = tf.data.Dataset.from_tensor_slices(t_data_in)
        dataset_in = dataset_in.map(input_map)

        dataset_out = tf.data.Dataset.from_tensor_slices(t_data_out)
        dataset_in = dataset_in.map(output_map)

        dataset = tf.data.Dataset.zip((dataset_in, dataset_out))

        t_batch_size = tf.placeholder(tf.int64, shape=[])
        dataset = dataset.batch(t_batch_size)
        dataset = dataset.repeat()
        if pp_map is not None:
            dataset = dataset.map(pp_map)
        if shuffle is not None:
            dataset = dataset.shuffle(shuffle)

        if self._t_handle is None:
            # We use a feedable iterator and not a reinitializable iterator,
            # as we don't want the training iterator to be reinitialized after
            # validation. Additionally, we can use initializable iterators that
            # way.
            self._t_handle = tf.placeholder(tf.string, shape=[])
            # FIXME: Find a better way than requiring a dataset to do this.
            data_iterator = tf.data.Iterator.from_string_handle(
                self._t_handle, dataset.output_types, dataset.output_shapes)
            #print(dataset.output_types)
            #print(dataset.output_shapes)

            next_element = data_iterator.get_next()
            self._t_ds_inputs = next_element[0]
            self._t_ds_outputs = next_element[1]

        # We cannot use a one shot iterator, as we haven't fed data into the
        # dataset yet.
        iterator = dataset.make_initializable_iterator()

        return iterator, t_batch_size

    def _build_datasets(self):
        """This method will create the training, validation and test dataset.

        It has to be called once before building the network.
        
        The following class members are set in this method (in addition to the
        once set in the method _make_dataset):
            self._t_train_raw_in: The raw training input data as received from
                                  the class Dataset.
            self._t_train_raw_out: The raw training output data as received
                                   from the class Dataset.
            self._train_iter: The iterator of the training dataset.
            self._t_train_batch_size: The placeholder for the training batch
                                      size.
            self._t_val_raw_in: The raw validation input data.
            self._t_val_raw_out: The raw validation output data.
            self._val_iter: The iterator of the validation dataset.
            self._t_val_batch_size: The placeholder for the validation batch
                                    size.
            self._t_test_raw_in: The raw test input data.
            self._t_test_raw_out: The raw test output data.
            self._test_iter: The iterator of the test dataset.
            self._t_test_batch_size: The placeholder for the test batch size.
        """
        d_in_size = np.prod(shared.data.in_shape)
        d_out_size = np.prod(shared.data.out_shape)

        # Initialize all 3 different datasets.
        if isinstance(shared.data, LargeImgDataset):
            # Inputs are strings (filenames).
            input_dtype = tf.string
            d_in_size = 1
        else:
            input_dtype = tf.float32

        self._t_train_raw_in = tf.placeholder(input_dtype,
            shape=[None, d_in_size])
        self._t_train_raw_out = tf.placeholder(tf.float32,
            shape=[None, d_out_size])
        self._train_iter, self._t_train_batch_size =  self._make_dataset( \
            self._t_train_raw_in, self._t_train_raw_out, mode='train',
            shuffle=500)

        self._t_val_raw_in = tf.placeholder(input_dtype,
            shape=[None, d_in_size])
        self._t_val_raw_out = tf.placeholder(tf.float32,
            shape=[None, d_out_size])
        self._val_iter, self._t_val_batch_size = self._make_dataset( \
                self._t_val_raw_in, self._t_val_raw_out)

        self._t_test_raw_in = tf.placeholder(input_dtype,
            shape=[None, d_in_size])
        self._t_test_raw_out = tf.placeholder(tf.float32,
            shape=[None, d_out_size])
        self._test_iter, self._t_test_batch_size = self._make_dataset( \
            self._t_test_raw_in, self._t_test_raw_out)

    def __del__ (self):
        """Cleanup"""
        if self._inference_session is not None:
            self._inference_session.close()

if __name__ == '__main__':
    pass


