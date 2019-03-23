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
@title           :config_exception.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/09/2018
@version         :1.0
@python_version  :3.6.6

A custom exception that may be raised if the user has wrongly used the config
file.

The implementation is based on an earlier implementation of a class I used in
another project:
    https://git.io/fNHoz

Note, the program only checks for obvious and unintended mistakes. There's no
comprehensive input check (i.e. a check that validates each parameter (type and
value). It is assumed that the user does not tamper with the options.
"""

import misc.shared_vars as shared

import logging
logger = logging.getLogger(shared.logging_name)

class ConfigException(Exception):
    """A special exception, that should only be raised in case the user has
    wrongly used the configuration file.

    Attributes:
    """
    def __init__(self, message):
        """Standard exception handling plus logging of the error message.

        Args:

        Returns:
        """
        super().__init__(message)

        # Log the error message, such that the error can be understood later
        # on.
        logger.critical(message)

if __name__ == '__main__':
    pass
