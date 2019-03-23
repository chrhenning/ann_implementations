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
@title           :examples/mi_toy_sample.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :09/26/2018
@version         :1.0
@python_version  :3.6.6

This provides an example interface to test Mutual Information Estimators.

Here are some results for different architecture grid searches:

Result summary using corr = 0.5:
  # Hidden Neurons 1 |   # Hidden Neurons 2 |   Mean Squared Error
---------------------|----------------------|---------------------
                  50 |                   25 |               0.0045
                  75 |                   38 |              0.00451
                 100 |                   50 |              0.00463
                  26 |                   13 |               0.0048
                   3 |                    2 |              0.01196
                   2 |                    2 |              0.01359
                   1 |                    1 |              0.01912
                   
Result summary using corr = 0.9:
  # Hidden Neurons 1 |   # Hidden Neurons 2 |   Mean Squared Error
---------------------|----------------------|---------------------
                   3 |                    2 |              0.03141
                  75 |                   38 |              0.03238
                  50 |                   25 |              0.03304
                  26 |                   13 |              0.03352
                 100 |                   50 |              0.03576
                   2 |                    2 |              0.07667
                   1 |                    1 |              0.41201
                   
Result summary  using corr = 0.0:
  # Hidden Neurons 1 |   # Hidden Neurons 2 |   Mean Squared Error
---------------------|----------------------|---------------------
                   1 |                    1 |                  0.0
                  75 |                   38 |                  0.0
                  50 |                   25 |                  0.0
                 100 |                   50 |                  0.0
                  26 |                   13 |                  0.0
                   2 |                    2 |                  0.0
                   3 |                    2 |                1e-05
"""

import time
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import os

from data.dataset import Dataset
import misc.shared_vars as shared
from networks.other.fc_mine import FCMINE
import configuration as config

import logging
logger = logging.getLogger(shared.logging_name)

class CorrBivariateGaussian(Dataset):
    """An instance of the class shall represent a dataset of mean-centerede
    correlated bivariate gaussian. The input vectors are thus 2D vectors drawn
    from the distribution. The output is a single scalar, denoting the mutual
    information between the input distributions (thus the same number for all
    samples).

    How one can compute the mutual information from such a distribution is
    described on slide 23 in the following distribution:
        https://gtas.unican.es/files/docencia/TICC/apuntes/tema1bwp_0.pdf

    Attributes: (additional to baseclass)
    """

    def __init__(self, N_train, N_test, N_val=0, std=1, corr=0.5):
        """Generate a random dataset.

        Args:
            N_train: Number of training samples in the dataset.
            N_test: Number of test samples in the dataset.
            N_val (default: 0): Number of validation samples in the dataset.
            std (default 1): Standard deviation of the two Gaussians.
            corr (default: 0.5): Correlation between the two Gaussians. Must be
                a value between [-1, 1].
        """
        super().__init__()

        start = time.time()
        print('Creating Correlated Bivariate Gaussian dataset ...')

        assert(corr >= -1 and corr <= 1)
        assert(N_train > 0 and N_test > 0)

        N = N_train + N_test + N_val

        var = std**2
        means = np.array([0, 0])
        cov = np.array([[var, corr*var], [corr*var, var]])

        samples = np.random.multivariate_normal(means, cov, size=N).astype( \
                np.float32)

        # Compute mutual information.
        mi = - 0.5 * np.log(1 - corr**2)

        self._data['classification'] = False
        self._data['sequence'] = False
        self._data['in_data'] = samples
        self._data['out_data'] = np.ones([N, 1], dtype=np.float32) * mi
        self._data['in_shape'] = [2]
        self._data['out_shape'] = [1]
        self._data['train_inds'] = np.arange(N_train)
        self._data['test_inds'] = np.arange(N_train, N_train + N_test)
        if N_val > 0:
            self._data['val_inds'] = np.arange(N_train + N_test, N)

        self._data['bigauss'] = dict()
        self._data['bigauss']['means'] = means
        self._data['bigauss']['cov'] = cov

        end = time.time()
        print('Elapsed time to read dataset: %f sec' % (end-start))

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'MI Correlated Bivariate Gaussians'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs=None, predictions=None):
        raise NotImplementedError('TODO implement')

ALLOW_PLOTS = True
NUM_ITER = 10001
# Only considered if architecture seach is performed.
CORR = 0.5
STD = 1
NUM_TRAIN = 100000
NUM_TEST = 1000
# Whether a grid search should be run over different architectures or over
# different correlation parameters.
ARCHITECTURE_SEARCH = False
# These two parameters are only considered if we are grid searching different
# correlation values.
H1_SIZE = 10
H2_SIZE = 5

def setup_network(allow_plots, layer_sizes=None, mode='inference'):
    """Setup the MINE network. Note, that this method will already build the
    network.

    Args:
        allow_plots: This option should only be enabled if a graphical backend
                     is available.
        layer_sizes (default: None): A list of integers, denoting the size of
                                     each hidden layer.
        mode: The network mode.

    Returns:
        The instantiated network.
    """
    net = FCMINE(mode=mode)
    net.allow_plots = allow_plots

    if layer_sizes is not None:
        for i, ls in enumerate(layer_sizes):
            net.set_layer_size(ls, i)

    net.build()

    return net

def search_architecture(num_train, num_test, allow_plots, num_iter, corr, std):
    """This method runs a simple grid search over a set of different
    architecture (different sizes of first and second hidden layer). Only a
    single correlation parameter is considered. Note, each network is only
    tested once, hence, this method only gives a coarse overview.

    Args:
        See keyword arguments of run method.
    """
    logger.info('### Creating dataset ...')

    data = CorrBivariateGaussian(num_train, num_test, std=std, corr=corr)

    # Important! Let the network know, which dataset to use.
    shared.data = data

    if allow_plots:
        r = 3*std
        ss = 2*r/101
        x, y = np.mgrid[-r:r:ss, -r:r:ss]
        pos = np.dstack((x, y))
        dist = multivariate_normal(data._data['bigauss']['means'],
                                   data._data['bigauss']['cov'])
        plt.title('The contour lines of the data distribution')
        plt.contourf(x, y, dist.pdf(pos))
        plt.show()

    # Construct a different dataset and test the run method and do the grid
    # search.
    data2 = CorrBivariateGaussian(2, 1000, std=std, corr=corr)
    real_mi = data2.next_test_batch(1)[1][0, 0]

    logger.info('### Creating dataset ... Done')

    logger.info('### Build, train and test network ...')

    # We will run a grid search over different layer sizes.
    num_configs = 5
    hidden_sizes_1 = np.round(np.linspace(100, 1, num_configs))
    hidden_sizes_2 = np.ceil(hidden_sizes_1 / 2).astype(np.int64).tolist()
    hidden_sizes_1 = hidden_sizes_1.astype(np.int64).tolist()
    # These two configs are interesting as well.
    hidden_sizes_1 += [2, 3]
    hidden_sizes_2 += [2, 2]

    out_folder = config.output_folder

    mse_values = []
    for i in range(len(hidden_sizes_1)):
        config.output_folder = os.path.join(out_folder, 'grid_%d' % i)
        if not os.path.exists(config.output_folder):
            os.mkdir(config.output_folder)

        size_h1 = hidden_sizes_1[i]
        size_h2 = hidden_sizes_2[i]
        logger.info('Running grid search step with hidden layer sizes: ' + \
                    '%d, %d.' % (size_h1, size_h2))

        train_net = setup_network(allow_plots, [size_h1, size_h2],
                                  mode='train')
        train_net.train(num_iter=num_iter)
    
        test_net = setup_network(allow_plots, [size_h1, size_h2],
                                 mode='inference')
        #test_net.test()
    
        samples = data2.get_test_inputs()
    
        estimated_mi = test_net.run(samples)
        logger.info('Estimated MI on additional dataset: %f' % estimated_mi)
        mse = (real_mi - estimated_mi)**2
        mse_values.append(mse)
        logger.info('### MSE on benchmark set: %f ###' % mse)

    logger.info('### Build, train and test network ... Done')

    config.output_folder = out_folder

    # Print results.
    logger.info('Result summary:')
    heading = ['# Hidden Neurons 1', '# Hidden Neurons 2',
               'Mean Squared Error']
    row_format = ' ' * 5 + "{:>20}" + " | {:>20}" * (len(heading) - 1)
    logger.info(row_format.format(*heading))
    logger.info(' ' * 5 + "-" * 20  + ("-|-" + "-" * 20) * (len(heading) - 1))
    order = np.argsort(mse_values).tolist()
    for i in order:
        elements = [hidden_sizes_1[i], hidden_sizes_2[i],
                    round(mse_values[i], 5)]
        logger.info(row_format.format(*elements))

    ind = int(np.argmin(mse_values))
    logger.info('### Best performing network is network %d with size:' % (ind)
                + ' %d, %d ###.' % (hidden_sizes_1[ind], hidden_sizes_2[ind]))

def run(**kwargs):
    """Run a toy example, estimating the Mutual Information of a Multivariate
    Gaussian using the MINE estimator.
    """
    allow_plots = ALLOW_PLOTS
    num_iter = NUM_ITER
    corr = CORR
    std = STD
    num_train = NUM_TRAIN
    num_test = NUM_TEST
    architecture_search = ARCHITECTURE_SEARCH
    h1_size = H1_SIZE
    h2_size = H2_SIZE

    for k in kwargs.keys():
        if k == 'allow_plots':
            allow_plots = kwargs[k]
        elif k == 'std':
            std = kwargs[k]
        elif k == 'corr':
            corr = kwargs[k]
        elif k == 'num_iter':
            num_iter = kwargs[k]
        elif k == 'num_train':
            num_train = kwargs[k]
        elif k == 'num_test':
            num_test = kwargs[k]
        elif k == 'architecture_search':
            architecture_search = kwargs[k]
        elif k == 'h1_size':
            h1_size = kwargs[k]
        elif k == 'h2_size':
            h2_size = kwargs[k]
        else:
            logger.warn('Keyword \'%s\' is unknown.' % k)

    if architecture_search:
        search_architecture(num_train, num_test, allow_plots, num_iter, corr,
                            std)
        return

    ## Run a grid search over different corr values.
    corr_vals = np.linspace(-0.9, 0.9, 12)

    estimated_mis = []
    real_mis = []

    out_folder = config.output_folder

    for i in range(len(corr_vals)):
        config.output_folder = os.path.join(out_folder, 'grid_%d' % i)
        if not os.path.exists(config.output_folder):
            os.mkdir(config.output_folder)

        corr = corr_vals[i]
        logger.info('###### Running grid search step with correlation: ' + \
                    '%f.' % (corr))

        logger.info('### Creating dataset ...')
        data = CorrBivariateGaussian(num_train, num_test, std=std, corr=corr)
        # Important! Let the network know, which dataset to use.
        shared.data = data

        real_mi = data.next_test_batch(1)[1][0, 0]

        logger.info('### Creating dataset ... Done')
    
        logger.info('### Build, train and test network ...')

        train_net = setup_network(allow_plots, [h1_size, h2_size],
                                  mode='train')
        train_net.train(num_iter=num_iter)
    
        test_net = setup_network(allow_plots, [h1_size, h2_size],
                                 mode='inference')
        #test_net.test()
    
        samples = data.get_test_inputs()
    
        estimated_mi = test_net.run(samples)
        logger.info('Estimated MI on test set: %f' % estimated_mi)
        mse = (real_mi - estimated_mi)**2
        logger.info('### MSE on test set: %f ###' % mse)

        estimated_mis.append(estimated_mi)
        real_mis.append(real_mi)

        logger.info('### Build, train and test network ... Done')

    config.output_folder = out_folder

    #plt.rcParams.update({'font.size': 22})
    #plt.figure(dpi=1200)

    # Plot results.
    plt.plot(corr_vals, real_mis, label='real')
    plt.plot(corr_vals, estimated_mis, label='estimated')
    plt.xlabel('Correlation')
    plt.ylabel('I(X;Z)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    pass


