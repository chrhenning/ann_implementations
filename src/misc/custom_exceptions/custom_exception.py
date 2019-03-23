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
@title           :custom_exception.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/10/2018
@version         :1.0
@python_version  :3.6.6

A custom exception that may be raised for any miscellaneous reason.
"""

import misc.shared_vars as shared

import logging
logger = logging.getLogger(shared.logging_name)

class CustomException(Exception):
    """A special exception, that logs the error after raising a normal
    exception.

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
