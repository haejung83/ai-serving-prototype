
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

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.preprocessing.image import Iterator
from tensorflow.python.keras._impl.keras.preprocessing.image import load_img
from tensorflow.python.keras._impl.keras.preprocessing.image import array_to_img
from tensorflow.python.keras._impl.keras.preprocessing.image import img_to_array

voc_classes = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError(
                'illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


class PascalVOCIterator(Iterator):

    def __init__(self,
                 directory,
                 set_name,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
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
        self.set_name = set_name
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

        self.classes = classes
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

        if not classes:
            classes = [key for key, value in voc_classes.items()]

        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        self.filenames = [l.strip().split(None, 1)[0] for l in open(
            os.path.join(directory, 'ImageSets', 'Main', set_name + '.txt')).readlines()]

        #pool = multiprocessing.pool.ThreadPool()

        self.samples = len(self.filenames)
        print('Found %d images belonging to %d classes.' % (self.samples,
                                                            self.num_classes))

        # second, build an index of the images
        self.classes = np.zeros((self.samples,), dtype='int32')

        for image_index in range(len(self.filenames)):
            boxes = self.load_annotations(image_index)
            if len(boxes) > 4:
                self.classes[image_index] = boxes[0, 4]

        #pool.close()
        #pool.join()
        super(PascalVOCIterator, self).__init__(
            self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) +
                           self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(
                os.path.join(self.directory, 'JPEGImages', fname + '.jpg'),
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

    def name_to_label_index(self, name):
        return self.class_indices[name]

    def __parse_annotation(self, element):
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name not in self.class_indices:
            raise ValueError('class name \'{}\' not found in class_indices: {}'.format(
                class_name, list(self.class_indices.keys())))

        box = np.zeros((1, 5))
        box[0, 4] = self.name_to_label_index(class_name)

        bndbox = _findNode(element, 'bndbox')
        box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box

    def __parse_annotations(self, xml_root):
        size_node = _findNode(xml_root, 'size')
        width = _findNode(size_node, 'width',  'size.width',  parse=float)
        height = _findNode(size_node, 'height', 'size.height', parse=float)

        boxes = np.zeros((0, 5))
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError(
                    'could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue
            boxes = np.append(boxes, box, axis=0)

        return boxes

    def load_annotations(self, image_index):
        filename = self.filenames[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(
                self.directory, 'Annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError(
                'invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError(
                'invalid annotations file: {}: {}'.format(filename, e)), None)

    def next(self):
        """For python 2.x.
        Returns:
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
