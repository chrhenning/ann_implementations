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
@title           :shared_vars.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/09/2018
@version         :1.0
@python_version  :3.6.6

A collection of global variables, that can be shared between modules.

I.e., each module can import variables in here and change them. Note, no thread
safety is enforced.

Note, if there are functions implemented in this module that modify a variable,
the keyword "global" has to be used at the beginning of the function.
"""

## Static variables, that should not be configured by the user.

# The name of the logger that is configured.
logging_name = 'ann_logger'

## Variables, that are once set (computed) and then used for the remaining
## runtime.

# An object of the class Dataset.
data = None

if __name__ == '__main__':
    pass


