from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
from six import raise_from

import multiprocessing.pool
import os
import re
import threading
import numpy as np

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.preprocessing.image import array_to_img
from tensorflow.python.keras.preprocessing.image import img_to_array


class ImageStoreGenerator(Iterator):

    def __init__(self,
                 directory,
                 image_store,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 interpolation='nearest',
                 skip_truncated=False,
                 skip_difficult=False,
                 ):

        if data_format is None:
            data_format = K.image_data_format()

        self.directory = directory
        self.image_store = image_store
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)

        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.classes = image_store.get_label_lut_list()

        if class_mode not in {'categorical', 'binary', 'sparse', 'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        # first, count the number of samples and classes
        self.samples = 0

        if not self.classes:
            raise ValueError("There is no classes, Please check given classes")

        self.num_classes = len(self.classes)

        # Collect filenames
        self.filenames = [image_label.get_image_name() 
            for image_label in self.image_store.get_image_label_list()]

        # A count of samples
        self.samples = len(self.filenames)
        print('Found %d images belonging to %d classes.' % (self.samples,
                                                            self.num_classes))

        # second, build an index of the images
        self.classes = np.zeros((self.samples,), dtype='int32')

        # Label LUT
        # label_lut = self.image_store.get_label_lut_list()

        for index, image_label in enumerate(self.image_store.get_image_label_list()):
            # Label index to description
            # self.classes[index] = label_lut[image_label.get_label()] 
            self.classes[index] = image_label.get_label()

        super(ImageStoreGenerator, self).__init__(
            self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) +
                           self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                grayscale=grayscale,
                target_size=self.target_size,
                interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        return batch_x, batch_y

    def next(self):
        print("next()")
        """For python 2.x.
        Returns:
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
