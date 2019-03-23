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
@title           :misc/visualizations/plotting_data.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :08/15/2018
@version         :1.0
@python_version  :3.6.6

A collections of functions that should support the plotting of data.

Why don't we use the plot_samples method of each dataset? For certain
architectures, the input output relations do not hold (e.g., for an
autoencoder, the output should be drawn from the input space). Capturing all
these cases in the plotting function of each class would be ugly. Instead, we
provide functions here, that try to be as general as possible, but very
informative about the qulity of network.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import misc.shared_vars as shared
from misc.custom_exceptions.custom_exception import CustomException

def plot_ae_images(title, sample_inputs, reconstructions, sample_outputs=None,
                   num_samples_per_row=3, show=True, filename=None,
                   interactive=False, figsize=(10, 6), outer_space=(0.4, 0.2),
                   inner_space=(0.2, 0.4)):
    """Plot input-output pairs of an autoencoder network.
    
    Args:
        title: The title of the whole figure.
        sample_inputs: A 2D numpy array, where each row is an input sample of a
            dataset, that encodes single images as inputs.
        reconstructions: The corresponding outputs of the Autoencoder network.
        sample_outputs (optional): The actual outputs according to the dataset.
            This is only used, if the dataset is a classification dataset, such
            that the class labels can be added to the titles.
        num_samples_per_row (default: 4): Maximum number of samples plotted
            per row in the generated figure.
        show (default: True): Whether the plot should be shown.
        filename (optional): If provided, the figure will be stored under
            this filename.
        interactive (default: False): Turn on interactive mode. We mainly
            use this option to ensure that the program will run in
            background while figure is displayed. The figure will be
            displayed until another one is displayed, the user closes it or
            the program has terminated. If this option is deactivated, the
            program will freeze until the user closes the figure.
            Note, if using the iPython inline backend, this option has no
            effect.
        figsize (default: (10, 6)): A tuple, determining the size of the
            figure in inches.
        outer_space (default: (0.4, 0.2)): A tuple. Defines the outer grid
            spacing for the plot (width, height).
        inner_space (default: (0.2, 0.2)): Same as outer_space, just for the
            inner grid space.
    """
    data = shared.data

    in_is_img, _ = data.is_image_dataset()

    if not in_is_img or data.sequence:
        raise CustomException('This method can only be called for datasets ' \
                              + 'with single images as inputs.')

    # Reverse one-hot encoding.
    if data.is_one_hot:
        if sample_outputs is not None and \
                sample_outputs.shape[1] == data.num_classes:
            sample_outputs = data._to_one_hot(sample_outputs, True)

    num_plots = sample_inputs.shape[0]
    num_cols = int(min(num_plots, num_samples_per_row))
    num_rows = int(np.ceil(num_plots / num_samples_per_row))

    fig = plt.figure(figsize=figsize)
    outer_grid = gridspec.GridSpec(num_rows, num_cols,
                                   wspace=outer_space[0],
                                   hspace=outer_space[1])

    # The 'y' is a dirty hack to ensure, that the titles are not overlapping.
    plt.suptitle(title, size=20, y=1.1)
    if interactive:
        plt.ion()

    for i in range(num_plots):
        inner_grid = gridspec.GridSpecFromSubplotSpec(1, 2,
            subplot_spec=outer_grid[i], wspace=inner_space[0],
            hspace=inner_space[1])

        subtitle = 'Sample %d' % i
        if data.classification:
            label = int(np.asscalar(sample_outputs[i]))
            subtitle += ' (Class %d)' % label

        ax = plt.Subplot(fig, outer_grid[i])
        ax.set_title(subtitle, fontsize=16)
        ax.set_axis_off()
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner_grid[0])
        ax.set_axis_off()
        ax.set_title('Original')
        ax.imshow(np.squeeze(np.reshape(sample_inputs[i, :], data.in_shape)))
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner_grid[1])
        ax.set_axis_off()
        ax.set_title('Reconstruction')
        ax.imshow(np.squeeze(np.reshape(reconstructions[i, :], data.in_shape)))
        fig.add_subplot(ax)

    if show:
        plt.show()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

def plot_gan_images(title, real_imgs, fake_imgs, real_outputs=None,
                    real_dis_outputs=None, fake_dis_outputs=None,
                    shuffle=False, num_samples_per_row=4, show=True,
                    filename=None, interactive=False, figsize=(10, 6),
                    spacing=(0.4, 0.4)):
    """Plot real and fake samples from a GAN network.
    
    Args:
        title: The title of the whole figure.
        real_imgs: A 2D numpy array, where each row is an input sample from a
            dataset, that encodes single images as inputs.
        fake_imgs: A 2D numpy array, where the images are generated by the
            generator network.
        real_outputs (optional): The actual outputs of the real images
            according to the dataset.
            This is only used, if the dataset is a classification dataset, such
            that the class labels can be added to the titles.
        real_dis_outputs (optional): The output confidences of the
            discriminator network for real images.
        fake_dis_outputs (optional): Same as "real_dis_outputs" for fake
            images.
        shuffle: Whether the order of images should be shuffled randomly.
        num_samples_per_row (default: 4): Maximum number of samples plotted
            per row in the generated figure.
        show (default: True): Whether the plot should be shown.
        filename (optional): If provided, the figure will be stored under
            this filename.
        interactive (default: False): Turn on interactive mode. We mainly
            use this option to ensure that the program will run in
            background while figure is displayed. The figure will be
            displayed until another one is displayed, the user closes it or
            the program has terminated. If this option is deactivated, the
            program will freeze until the user closes the figure.
            Note, if using the iPython inline backend, this option has no
            effect.
        figsize (default: (10, 6)): A tuple, determining the size of the
            figure in inches.
        spacing (default: (0.2, 0.2)): A tuple. Defines the spacing between
            subplots (width, height).
    """
    data = shared.data

    in_is_img, _ = data.is_image_dataset()

    if not in_is_img or data.sequence:
        raise CustomException('This method can only be called for datasets ' \
                              + 'with single images as inputs.')

    # Reverse one-hot encoding.
    if data.is_one_hot:
        if real_outputs is not None and \
                real_outputs.shape[1] == data.num_classes:
            real_outputs = data._to_one_hot(real_outputs, True)

    num_plots = real_imgs.shape[0] + fake_imgs.shape[0]
    num_cols = int(min(num_plots, num_samples_per_row))
    num_rows = int(np.ceil(num_plots / num_samples_per_row))

    real_or_fake = np.concatenate([np.ones(real_imgs.shape[0]),
                                   np.zeros(fake_imgs.shape[0])])
    if shuffle:
        np.random.shuffle(real_or_fake)
    real_ind = 0
    fake_ind = 0

    fig = plt.figure(figsize=figsize)

    plt.suptitle(title, size=20)
    plt.subplots_adjust(wspace=spacing[0], hspace=spacing[1])

    if interactive:
        plt.ion()

    for i in range(num_plots):
        ax = fig.add_subplot(num_rows, num_cols, i+1)

        if real_or_fake[i]:
            img = real_imgs[real_ind, :]
            if real_outputs is None or not data.classification:
                subtitle = 'Real'
            else:
                label = int(np.asscalar(real_outputs[real_ind]))
                subtitle = 'Real (Class %d)' % (label)
            if real_dis_outputs is not None:
                subtitle += '\nDiscriminator: %.4f' % \
                    round(np.asscalar(real_dis_outputs[real_ind]), 4)
            real_ind += 1
        else:
            img = fake_imgs[fake_ind, :]
            subtitle = 'Fake'
            if fake_dis_outputs is not None:
                subtitle += '\nDiscriminator: %.4f' % \
                    round(np.asscalar(fake_dis_outputs[fake_ind]), 4)
            fake_ind += 1

        ax.set_axis_off()
        ax.set_title(subtitle)
        ax.imshow(np.squeeze(np.reshape(img, data.in_shape)))

    if show:
        plt.show()

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

if __name__ == '__main__':
    pass


