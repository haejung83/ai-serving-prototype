import os
import sys
import glob
import argparse

from tensorflow.python.keras import __version__
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
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
from tensorflow.python.keras._impl.keras.engine.network import get_source_inputs
from tensorflow.python.keras._impl.keras.engine import saving
from tensorflow.python.keras._impl.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras._impl.keras import backend as K

from tensorflow.python.keras._impl.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.keras._impl.keras.applications import mobilenet

# ThinModel
from thin_model import ThinModel
from thin_model import thin_model_from_json

import h5py

# Constants
IM_WIDTH, IM_HEIGHT = 224, 224
NB_EPOCHS = 20
BATCH_SIZE = 20
   

def setup_to_transfer_learn(model, base_model):
    # Freeze all of layer on base model
    for layer in base_model.layers:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in model.layers:
        print(layer.name, layer.trainable)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(1e-3, name='dropout')(x)
    x = Conv2D(nb_classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((nb_classes,), name='reshape_2')(x)

    model = ThinModel(inputs=base_model.input, outputs=x)
    model.set_separation_mark(base_model.layers[-1])
    model.summary()

    print("Base Model Layers: %s" % (len(base_model.layers)))
    print("Merged Model Layers: %s" % (len(model.layers)))
    return model


def train(args):
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
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = add_new_last_layer(base_model, nb_classes)

    if num_gpus > 1:
        gpu_model = multi_gpu_model(model, gpus=num_gpus)
    else:
        gpu_model = model

    # make model to be transfer learning
    setup_to_transfer_learn(gpu_model, base_model)

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

    # Mobilenet only
    with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
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
    a.add_argument("--batch_size", default=BATCH_SIZE)
    a.add_argument("--output_model_name", default="mobilenet_model")
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
