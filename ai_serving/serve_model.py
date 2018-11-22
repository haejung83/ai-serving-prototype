from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.network import get_source_inputs
from tensorflow.python.keras.engine import saving
from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras import backend as K

from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.keras.applications import mobilenet

from ODM.odm_serve_model import ODMServeModel, create_odm_serve_model
from generator.image_store_generator import ImageStoreGenerator
from thin_model import ThinModel
from thin_model import thin_model_from_json
import ai_config as config

from enum import IntEnum
from PIL import Image as PILImage
import numpy as np
import h5py
import os


class ServeModelType(IntEnum):
    Generic = 1
    Animal = 2
    Food = 3
    Infrastructure = 4


class ServeModel(object):
    _target_size = (224, 224)

    def __init__(self, odm):
        self._odm = odm
        self._path = None
        self._base_model = None
        self._thin_model = None
        self._is_built = False
        self._data_generator = dict()
        self._make_essential()

    @staticmethod
    def create_from_odm(odm_serve_model):
        return ServeModel(odm_serve_model)

    @staticmethod
    def delete_by_odm(odm_serve_model):
        if isinstance(odm_serve_model, ODMServeModel):
            odm_serve_model.delete()
        else:
            raise ValueError('[%s] %s' %(__name__,
                "The object is not ODMServeModel instnace. So couldn't delete it"))

    @staticmethod
    def create(
            hash_key,
            model_type,
            architecture_path=None,
            freezed_weight_path=None,
            tuned_weight_path=None):
        return ServeModel.create_from_odm(
            create_odm_serve_model(
                hash_key=hash_key,
                model_type=model_type,
                architecture_path=architecture_path,
                freezed_weight_path=freezed_weight_path,
                tuned_weight_path=tuned_weight_path))

    def _make_essential(self):
        _serve_model_path = os.path.join(
            config.SERVING_PROPERTIES[config.SERVING_PROP_PROJECT_ROOT],
            self._odm.hash_key,
            config.SERVING_PROPERTIES[config.SERVING_PROP_SERVE_MODEL_PATH])

        if not os.path.exists(_serve_model_path):
            os.makedirs(_serve_model_path)
            print('[%s] %s' %(__name__,
                'Make dir ' + _serve_model_path))

        self._path = _serve_model_path

    def load(self):
        # Mobilenet only
        with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
            self._thin_model = thin_model_from_json(
                root_path=self._path,
                filename=self._odm.hash_key
            )
        print('[%s] Loaded ThinModel %s' % (__name__, self._thin_model))

        self._is_built = True

    def save(self):
        if not self._thin_model:
            raise ValueError('[%s] %s' %(__name__,
                "There is no exist ThinModel. Can't save it"))

        _arch, _freezed, _tuned = self._thin_model.save_weights_separately(
            root_path=self._path,
            filename=self._odm.hash_key
        )

        self._odm.architecture_path = _arch
        self._odm.freezed_weight_path = _freezed
        self._odm.tuned_weight_path = _tuned
        self._odm.save()

    def train(self, image_store, **kwargs):
        if not self._is_built:
            self._validate_image_store(image_store)
            self._network_build()
            self._make_network_trainable()
            self._prepare_datagenerator(image_store)
        
        self._thin_model.fit_generator(
            generator=self._data_generator["train"],
            epochs=10,
            workers=2,
            validation_data=self._data_generator["validation"]
        )

    def predict(self, image_store, image_name=None,
                image_index=None, image_data=None, **kwargs):
        if not self._is_built:
            self.load()

        self._validate_image_store(image_store)

        if image_data:
            # Predict with a given raw image
            _prediction_result = self._thin_model.predict(image_data)
        elif image_name or image_index is not None:
            # Predict with a specific image 
            _loaded_image = self._get_image_from_image_store(
                image_store=image_store,
                image_name=image_name,
                image_index=image_index)
            _prediction_result = self._thin_model.predict(_loaded_image)
        else:
            raise ValueError('There is no method to predict with a given image data')

        _prediction_result = np.squeeze(_prediction_result)
        if 'desc' in kwargs and kwargs['desc']:
            _prediction_result = image_store.get_label_description(
                label_index=np.argmax(_prediction_result)
                )

        # Call the callback for passing result
        if 'callback' in kwargs:
            kwargs['callback'](_prediction_result)

    def _get_image_from_image_store(self, image_store, image_name=None, image_index=None):
        _filename = image_store.get_image_path(image_name=image_name, image_index=image_index)
        if not _filename:
            raise ValueError('There is no image data to load from the ImageStore')

        _loaded_image = self._get_image_from_file(_filename)
        _loaded_image = self._preprocess_image(_loaded_image)

        return _loaded_image

    def _get_image_from_file(self, filename):
        return PILImage.open(filename)

    def _preprocess_image(self, raw_image):
        if raw_image.size != ServeModel._target_size:
            raw_image = raw_image.resize(ServeModel._target_size)

        _preprocessed_image  = img_to_array(raw_image)
        _preprocessed_image = np.expand_dims(_preprocessed_image, axis=0)
        _preprocessed_image = preprocess_input(_preprocessed_image)

        return _preprocessed_image

    def get_score(self, **kwargs):
        if not self._is_built:
            self.load()

        raise NotImplementedError(
            '[%s] get_score method not implemented yet' % (__name__))

    # Build own network lazily
    def _network_build(self):
        num_classes = self._odm.num_classes

        if num_classes == 0:
            raise ValueError("[%s] This network has not any classification class" % (__name__))

        _base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Add new layers
        x = _base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 1024), name='reshape_1')(x)
        x = Dropout(1e-3, name='dropout')(x)
        x = Conv2D(num_classes, (1, 1), padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((num_classes,), name='reshape_2')(x)

        # Build a ThinModel
        self._thin_model = ThinModel(inputs=_base_model.input, outputs=x)
        self._thin_model.set_separation_mark(_base_model.layers[-1])
        self._thin_model.summary()

        self._base_model = _base_model

        self._is_built = True

    def _make_network_trainable(self):
        for layer in self._base_model.layers:
            layer.trainable = False

        self._thin_model.compile(
            optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def _validate_image_store(self, image_store):
        if not image_store:
            raise ValueError("[%s] The ImageStore is not availiable" % (__name__))

        label_lut_list =  image_store.get_label_lut_list()
        if not label_lut_list:
            raise ValueError("[%s] The ImageStore has not label lut list" % (__name__))

        num_classes = len(label_lut_list)
        if num_classes == 0:
            raise ValueError("[%s] The ImageStore has not any class" % (__name__))

        if self._odm.num_classes != num_classes:
            self._odm.num_classes = num_classes
            self._odm.save()

    def _prepare_datagenerator(self, image_store):
        train_dir = image_store.get_path()
        val_dir = image_store.get_path()

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        self._data_generator["train"] = ImageStoreGenerator(
            directory=train_dir,
            image_store=image_store,
            image_data_generator=train_datagen,
            target_size=(224, 224),
            batch_size=2
        )

        self._data_generator["validation"] = ImageStoreGenerator(
            directory=val_dir,
            image_store=image_store,
            image_data_generator=train_datagen,
            target_size=(224, 224),
            batch_size=2
        )
