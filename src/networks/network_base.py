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
@title           :networks/network_base.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/09/2018
@version         :1.0
@python_version  :3.6.6

An abstract base class for networks implemented in this framework.
"""

from abc import ABC, abstractmethod
from misc.custom_exceptions.argument_exception import ArgumentException

class NetworkBase(ABC):
    """An interface for implemented networks. If the networks are adhering the
    rules of this interface, they are easy to integrate in the whole framework.

    Attributes:
        mode: Is this network running in training or inference mode. This
            distinction is important, as it might change the way the network is
            build. The attribute is a string with the valid values 'train' and
            'inference'.
        continue_training: If implemented by the instantiated network, this
            option can be used to continue a recently aborted training.
            Hence, if possible, the network should attempt to start the
            training where it has ended before instead of starting from
            scratch.
        allow_plots: If the code is running on a machine without windowing
            system (such as an SSH connection to a host without X11 running),
            then we don't want the networks to plot visualizations (other
            channels, such as Tensorboard, can still be used to evaluate the
            network).
    """
    def __init__(self, mode='train'):
        """This super constructor should be called by all implementations of
        the interface.

        Args:
            mode: The value of the attribute "mode".
        """
        if mode != 'train' and mode != 'inference':
            raise ArgumentException('The network mode "' + mode + '" is ' +
                                    'invalid. It has to be "train" or ' +
                                    '"inference"')
        self._mode = mode

        # This attribute should be set true in the build method.
        self._is_build = False

        self._continue_training = False

        self._allow_plots = False

    @property
    def mode(self):
        """Getter for read-only attribute mode."""
        return self._mode

    @property
    def continue_training(self):
        """Getter for the attribute continue_training."""
        return self._continue_training

    @continue_training.setter
    def continue_training(self, value):
        """Setter for the attribute continue_training."""
        self._continue_training = value

    @property
    def allow_plots(self):
        """Getter for the attribute allow_plots."""
        return self._allow_plots

    @allow_plots.setter
    def allow_plots(self, value):
        """Setter for the attribute allow_plots."""
        self._allow_plots = value

    def is_training(self):
        """Is the network in training mode.

        Returns:
            Returns True, if the network mode is "train".
        """
        return self._mode == "train"

    @abstractmethod
    def build(self):
        """Build the network, such that we can use it to run training or
        inference."""
        # Note, in implementations of this method, you should add the line:
        # self._is_build = True
        pass

    @abstractmethod
    def train(self):
        """Train the network."""
        pass

    @abstractmethod
    def test(self):
        """Evaluate the trained network using the whole test set."""
        pass

    @abstractmethod
    def run(self, inputs):
        """Run the network with the given inputs.

        Args:
            inputs: Samples that align with the dataset (2D numpy array).

        Returns:
            The outputs of the network as 2D numpy array.
        """
        pass

if __name__ == '__main__':
    pass


