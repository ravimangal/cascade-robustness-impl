import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model, Sequential

def arch_safescad(input_shape=None,
          initializers='he_normal',
          classes=5):

    if input_shape is None:
        raise ValueError(
            "The shape of the input layer must be not be `None`."
        )

    '''
        Model architecture:
        Input Neurons: 25(Number of features in each data point)
        Hidden Layers: Dense layers with 50-100-35-11
        Output Neurons: 5 (Number of classes)
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape)))
    model.add(Dense(50, activation='relu', kernel_initializer=initializers))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(35, activation='relu'))
    model.add(Dense(11, activation='relu'))
    #    model.add(Dense(classes, activation='softmax'))  # logits layer
    model.add(Dense(classes))  # logits layer
    return model

def arch_safescad2(input_shape=None,
          initializers='he_normal',
          classes=3):

    if input_shape is None:
        raise ValueError(
            "The shape of the input layer must be not be `None`."
        )

    '''
        Model architecture:
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=(input_shape)))
    model.add(Dense(23, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Reshape((80, 1)))
    model.add(MaxPool1D(pool_size=4, strides=1, padding='valid'))
    model.add(Reshape((77,)))
    model.add(Dense(40, activation='relu'))
    model.add(Reshape((40, 1)))
    model.add(MaxPool1D(pool_size=4, strides=1, padding='valid'))
    model.add(Reshape((37,)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(classes))
    return model

def arch_safescad_original(input_shape=None,
          initializers='he_normal',
          classes=5):

    if input_shape is None:
        raise ValueError(
            "The shape of the input layer must be not be `None`."
        )

    # v3.2.2
    # loss: 0.3316 - accuracy: 0.8707 - val_loss: 0.3212 - val_accuracy: 0.874
    # 1) Train: 0.869, 2) Test: 0.847
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(23, activation='relu', kernel_initializer=initializers))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(11, activation='relu'))
    # model.add(Dense(classes, activation='softmax'))  # logits layer
    model.add(Dense(classes))  # logits layer
    return model