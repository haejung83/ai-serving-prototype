# Transfer Learning Test
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

import h5py


# Constants for Xception Network Model
IM_WIDTH, IM_HEIGHT = 224, 224
NB_EPOCHS = 20
BAT_SIZE = 20
FC_SIZE = 2048


def relu6(x):
    return K.relu(x, max_value=6)

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
    x = Activation('relu', name='act_4')(x)
  
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
 
    model = Model(inputs, x, name='T1000_Net' )

    # TODO: Load weight if it has
    return model
    

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False

        # Check the trainable status of the individual layers
    for layer in model.layers:
        print(layer, layer.trainable)

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet

    Args:
      base_model: keras model excluding top
      nb_classes: # of classes

    Returns:
      new keras model with last layer
    """
    # x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Dense(classes, activation='softmax', name='predictions')(x)

    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    x = Dropout(1e-3)(x)
    x = Dense(nb_classes * 2, activation='relu', name='addional_dense')(x)

    predictions = Dense(nb_classes, activation='softmax',
                        name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    print("Base Model Layers: %s" % (len(base_model.layers)))
    print("Merged Model Layers: %s" % (len(model.layers)))
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
    #base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #model = add_new_last_layer(base_model, nb_classes)
    model = T1000(input_shape=(224, 224, 3),weights='imagenet', include_top=True)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    if num_gpus > 1:
        gpu_model = multi_gpu_model(model, gpus=num_gpus)
    else:
        gpu_model = model

    # make model to be transfer learning
    # setup_to_transfer_learn(gpu_model, base_model)

    history = gpu_model.fit_generator(
        train_generator,
        epochs=nb_epoch,
        workers=2,
        validation_data=validation_generator)

    # save_and_load_test(model)
    model_save(model, args)

    score = gpu_model.evaluate_generator(generator=validation_generator)
    print('Eval score:', score[0])
    print('Eval accuracy:', score[1])


def model_save(model, args):
    save_model_name = args.output_model_name

    # svae model structure with json
    model_json = model.to_json()
    with open(save_model_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    #model.save_weights(save_model_name + ".h5")
    #print("Saved model to disk")

    # freezed layers
    freezed_layers = model.layers[:-6]
    with h5py.File(save_model_name + ".h5", 'w') as f:
      saving.save_weights_to_hdf5_group(f, freezed_layers)

    # extract dense layers and save it    
    fine_tuned_layers = model.layers[-5:]
    with h5py.File('fine_tuned_layers.h5', 'w') as f:
      saving.save_weights_to_hdf5_group(f, fine_tuned_layers)

    # For comparison
    model.save_weights("whole_weights.h5")

    for layer in fine_tuned_layers:
        print(layer)

    # Extra Testing
    #seperate_layer = model.layers[-6]
    #print(seperate_layer)
    #denseInput = Input(tensor=seperate_layer, shape=seperate_layer.shape)
    #denseInput = Input(tensor=seperate_layer)
    #new_model = Model(denseInput, model.layers[-1])
    #new_model.summary()


def load_and_evaluate(args):
    load_model_name = args.output_model_name
    batch_size = int(args.batch_size)
    num_gpus = int(args.gpus)

    # load json and create model
    json_file = open(load_model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights(load_model_name + ".h5")
    loaded_model.load_weights(load_model_name + ".h5", by_name=True)

    loaded_model.load_weights('fine_tuned_layers.h5', by_name=True)
    print("Loaded model from disk")

    if num_gpus > 1:
        loaded_model = multi_gpu_model(loaded_model, gpus=num_gpus)

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])

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

    predict_result = loaded_model.predict_generator(generator=test_generator)

    for presult in predict_result:
        print('Predict result: ', presult)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_name", default="xception_model")
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
