import os
import sys
import glob
import argparse

from tensorflow.python.keras import __version__
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.xception import preprocess_input
from tensorflow.python.keras.applications import Xception
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
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras import regularizers
from tensorflow.python.keras._impl.keras.engine.network import get_source_inputs
from tensorflow.python.keras._impl.keras.engine import saving
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras import backend as K

# ThinModel
from thin_model import ThinModel
from thin_model import thin_model_from_json

import h5py

IM_WIDTH, IM_HEIGHT = 224, 224
NB_EPOCHS = 20
BAT_SIZE = 20

# Custom Activatioin Function (Relu, Limited until 6)
def relu6(x):
    return K.relu(x, max_value=6)

# T1000 Net (Custom, Referenced Xception Network)
def T1000(input_shape=None,
          dropout=1e-3,
          include_top=True,
          input_tensor=None,
          weights='imagenet',
          classes=3):

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if K.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if input_tensor is None:
       img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Convolution 1
    x = Conv2D(
        64,
        (1,1),
        padding='same',
        use_bias='false',
        name='conv_1')(img_input)
    x = BatchNormalization(axis=channel_axis, name='bn_1')(x)
    x = Activation('relu', name='act_1')(x)

    # Convolution 2
    x = Conv2D(
        64,
        (3,3),
        strides=(2, 2),
        padding='same',
        use_bias='false',
        name='conv_2')(x)
    x = BatchNormalization(axis=channel_axis, name='bn_2')(x)
    x = Activation('relu', name='act_2')(x)

    # Convolution 3
    x = Conv2D(
        128,
        (1,1),
        padding='same',
        use_bias='false',
        name='conv_3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn_3')(x)
    x = Activation('relu', name='act_3')(x)
 
    # Convolution 4
    x = Conv2D(
        128,
        (3,3),
        strides=(2, 2),
        padding='same',
        use_bias='false',
        name='conv_4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn_4')(x)

    mark_layer = Activation('relu', name='act_4')
    x = mark_layer(x)
  
    # Dense Layers 
    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu', use_bias=True, name='dense_1')(x)
        x = Dropout(dropout)(x)
        x = Dense(classes, activation='softmax', name='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
 
    # Testing the ThinModel
    model = ThinModel(inputs, x, name='T1000_Net')
    
    # Mark the separation layer
    model.set_separation_mark(mark_layer)

    # TODO(haejung): Load weight if it has
    return model


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)
    num_gpus = int(args.gpus)

    # data prep
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    # setup model, include_top=False excludes final FC layer
    model = T1000(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
    model.summary()

    if num_gpus > 1:
        gpu_model = multi_gpu_model(model, gpus=num_gpus)
    else:
        gpu_model = model

    gpu_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    history = gpu_model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        workers=2,
        validation_data=validation_generator)

    separation_save(model, args)

    score = gpu_model.evaluate_generator(generator=validation_generator)
    print('Eval score:', score[0])
    print('Eval accuracy:', score[1])


def separation_save(model, args):
    load_model_name = args.output_model_name
    print('Save model to disk: ' + load_model_name)
    model.save_weights_separately(load_model_name)


def load_and_evaluate(args):
    load_model_name = args.output_model_name
    batch_size = int(args.batch_size)
    num_gpus = int(args.gpus)

    loaded_model = thin_model_from_json(load_model_name)
    print("Loaded model from disk")

    if num_gpus > 1:
        loaded_model = multi_gpu_model(loaded_model, gpus=num_gpus)

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])

    print("Compiled the loaded model")

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    score = loaded_model.evaluate_generator(generator=test_generator)
    print('Eval score:', score[0])
    print('Eval accuracy:', score[1])

    # predict_result = loaded_model.predict_generator(generator=test_generator)
    # for presult in predict_result:
        # print('Predict result: ', presult)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_name", default="t1000_model")
    a.add_argument("--gpus", default=0)
    a.add_argument("--mode", default="train")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    if args.mode == 'train':
        train(args)
    else:
        load_and_evaluate(args)
