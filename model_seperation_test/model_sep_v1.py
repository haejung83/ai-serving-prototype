from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import Xception
from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.utils import multi_gpu_model

from tensorflow.python.keras.models import load_model

import os
import numpy as np
import matplotlib.pyplot as plt

# conv_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base.summary()

#train_dir = './train_cards'
#validation_dir = './train_cards'
#nTrain = 100
#nVal = 100
#nOutputLayerDepth = 2048

train_dir = './train'
validation_dir = './validation'
nTrain = 800
nVal = 200
nOutputLayerDepth = 1024

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20
 
def gen_feature_label(image_dir, image_count): 
    gen_features = np.zeros(shape=(image_count, 7, 7, nOutputLayerDepth))
    gen_labels = np.zeros(shape=(image_count, 3))

    train_generator = datagen.flow_from_directory(
        image_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = conv_base.predict(inputs_batch)
        gen_features[i * batch_size : (i + 1) * batch_size] = features_batch
        gen_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= image_count:
            break

    return gen_features, gen_labels


def train_and_save_model(in_epochs=20):
    train_features, train_labels = gen_feature_label(train_dir, nTrain)
    #train_features = np.reshape(train_features, (nTrain, 7 * 7 * nOutputLayerDepth))
    #train_features = np.reshape(train_features, (nTrain, 7, 7, nOutputLayerDepth))

    print(train_features.shape)
    print(train_labels.shape)

    validation_features, validation_labels = gen_feature_label(validation_dir, nVal)
    #validation_features = np.reshape(validation_features, (nVal, 7 * 7 * nOutputLayerDepth))
    #validation_features = np.reshape(validation_features, (nVal, 7, 7, nOutputLayerDepth))

    print(validation_features.shape)
    print(validation_labels.shape)

    # x = GlobalAveragePooling2D()(x)
    # x = Reshape(shape, name='reshape_1')(x)
    # x = Dropout(dropout, name='dropout')(x)
    # x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    # x = Activation('softmax', name='act_softmax')(x)
    # x = Reshape((classes,), name='reshape_2')(x)
    
    model = models.Sequential()
    #model.add(layers.Dense(32, activation='relu', input_dim=7 * 7 * nOutputLayerDepth))
    #model.add(layers.Conv2D(1024, (1,1), padding='same', input_shape=(7, 7, nOutputLayerDepth)))
    model.add(layers.GlobalAveragePooling2D(input_shape=(7, 7, nOutputLayerDepth)))
    model.add(layers.Reshape(target_shape=(1, 1, 1024)))
    model.add(layers.Dropout(1e-3))
    model.add(layers.Conv2D(3, (1,1), padding='same', name='conv_preds'))
    model.add(layers.Dense(3, activation='softmax'))
    model.add(layers.Flatten())

    # model = multi_gpu_model(model, gpus=2)
    model.summary()

    model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                loss='categorical_crossentropy',
                metrics=['acc'])
    
    history = model.fit(train_features,
                        train_labels,
                        epochs=in_epochs,
                        batch_size=batch_size,
                        validation_data=(validation_features, validation_labels))

    model.save('./trained_model.h5', include_optimizer=False)
    print(model.to_json())
    
    result_predict = model.predict(validation_features[0:1])
    print("Result of Predict %s" %(result_predict))

def load_and_predict(in_weight_path, in_target_path):
    model = load_model(in_weight_path)
    model.summary()

    validation_features, validation_labels = gen_feature_label(validation_dir, nVal)
    validation_features = np.reshape(validation_features, (nVal, 7 * 7 * nOutputLayerDepth))

    result_predict = model.predict(validation_features[0:1])
    print("Result of Predict %s" %(result_predict))

'''
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('chart.png')
#plt.show()
'''

if __name__ == '__main__':
    train_and_save_model(100)
    # load_and_predict('./trained_model.h5', '')