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
@title           :data/large_img_dataset.py
@author          :ch
@contact         :henningc@ethz.ch
@created         :09/20/2018
@version         :1.0
@python_version  :3.6.6

This class is an abstract wrapper for large datasets, that have images as
inputs. Typically, these datasets are too large to be loaded into memory.
Though, their outputs (labels) can still easily be hold in memory. Hence, the
idea is, that instead of loading the actual images, we load the paths for each
image into memory. Then we can load the images from disk as needed.

To sum up, handlers that implement this interface will hold the outputs and
paths for the input images of the whole dataset in memory, but not the actual
images.

As an alternative, one can implement wrappers for HDF5 and TFRecord files.

Here is a simple example that illustrates the format of the dataset:
    https://www.tensorflow.org/guide/datasets#decoding_image_data_and_resizing_it
"""
import numpy as np
import os
import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image

from data.dataset import Dataset

class LargeImgDataset(Dataset):
    """A general dataset template for datasets with images as inputs, that are
    locally stored as individual files. Note, that this is an abstract class
    that should not be instantiated.

    Hints, when implementing the interface.
        - Attribute 'in_shape' still has to be correctly implemented,
          independent of the fact, that the actual input data is a list of
          strings.

    Attributes (additional to base class):
        imgs_path: The base path of all images.
        png_format_used: Whether png or jped encoded of images is assumed.
    """
    def __init__(self, imgs_path, png_format=False):
        """Initialize internal structure.

        Args:
            imgs_path: The path to the folder, containing the image files.
                       (the actual image paths contained in the input data will
                       be concatenated to this path).
            png_format (default: False): The images are typically assumed to be
                                         jpeg encoded. You may change this to
                                         png enocded images.
        """
        super().__init__()

        self._imgs_path = imgs_path
        self._png_format_used = png_format

        # The wrapper is currently not meant for sequence inputs. You can still
        # set this variable to true, if you have sequence outputs.
        self._data['sequence'] = False

    @property
    def imgs_path(self):
        """Getter for read-only attribute imgs_path"""
        return self._imgs_path

    @property
    def png_format_used(self):
        """Getter for read-only attribute png_format_used"""
        return self._png_format_used

    def get_train_inputs(self):
        """Get the inputs of all training samples.
        
        Returns:
            An np.chararray, where each row corresponds to an image file name.
        """
        return Dataset.get_train_inputs(self)

    def get_test_inputs(self):
        """Get the inputs of all test samples.
        
        Returns:
            An np.chararray, where each row corresponds to an image file name.
        """
        return Dataset.get_test_inputs(self)

    def get_val_inputs(self):
        """Get the inputs of all validation samples.
        
        Returns:
            An np.chararray, where each row corresponds to an image file name.
        """
        return Dataset.get_val_inputs(self)

    def read_images(self, inputs):
        """For the given filenames, read and return the images.

        Args:
            inputs: An np.chararray of filenames.

        Returns:
            A 2D numpy array, where each row contains a picture.
        """
        ret = np.empty([inputs.shape[0], np.prod(self.in_shape)], np.float32)

        for i in range(inputs.shape[0]):
            fn = os.path.join(self.imgs_path, 
                              str(inputs[i, np.newaxis].squeeze()))
            img = Image.open(fn)
            #img = mpimg.imread(fn)
            img = img.resize(self.in_shape[:2], Image.BILINEAR)
            ret[i, :] = np.array(img).flatten()

        # Note, the images already have pixel values between 0 and 1 for
        # PNG images.
        if not self.png_format_used:
            ret /= 255.

        return ret

    def tf_input_map(self, mode='inference'):
        """This method should be used by the map function of the Tensorflow
        Dataset interface (tf.data.Dataset.map). In the default case, this is
        just an identity map, as the data is already in memory.

        There might be cases, in which the full dataset is too large for the
        working memory, and therefore the data currently needed by Tensorflow
        has to be loaded from disk. This function should be used as an
        interface for this process.

        Args:
            mode: This is the same as the mode attribute in the class
                  NetworkBase, that can be used to distinguish between training
                  and inference (e.g., if special input processing should be
                  used during training).

        Returns:
            A function handle, that maps the given input tensor to the
            preprocessed input tensor.
        """
        base_path = os.path.join(self.imgs_path, '')
        
        def load_inputs(inputs):
            filename = tf.add(base_path, tf.squeeze(inputs))
            image_string = tf.read_file(filename)
            if self.png_format_used:
                image = tf.image.decode_png(image_string)
            else:
                image = tf.image.decode_jpeg(image_string)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize_images(image, self.in_shape[:2])
            # We always feed flattened images into the network.
            image = tf.reshape(image, [-1])

            return image
        
        return load_inputs


if __name__ == '__main__':
    pass


