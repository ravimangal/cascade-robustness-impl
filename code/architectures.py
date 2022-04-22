import tensorflow.keras.backend as K
import tensorflow as tf

from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model, Sequential

#from deel.lip.initializers import BjorckInitializer
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

from gloro.layers import InvertibleDownsampling as InvertibleDownSampling
from gloro.layers import MinMax
from gloro.layers import ResnetBlock


def _get_initializer(initialization):
    if initialization == 'orthogonal':
        return Orthogonal()

    elif initialization == 'glorot_uniform':
        return GlorotUniform()

    # elif initialization == 'bjorck':
    #     return BjorckInitializer()

    else:
        raise ValueError(f'unknown initialization: {initialization}')

def _add_pool(z, pooling_type, activation=None, initializer=None):
    if pooling_type == 'avg':
        return AveragePooling2D()(z)

    elif pooling_type == 'conv':
        channels = K.int_shape(z)[-1]

        if initializer is None:
            initializer = _get_initializer('orthogonal')

        z = Conv2D(
            channels,
            4,
            strides=2,
            padding='same',
            kernel_initializer=initializer)(z)

        return _add_activation(z, activation)

    elif pooling_type == 'invertible':
        return InvertibleDownSampling()(z)

    else:
        raise ValueError(f'unknown pooling type: {pooling_type}')

def _add_activation(z, activation_type='relu'):
    if activation_type == 'relu':
        return Activation('relu')(z)

    elif activation_type == 'elu':
        return Activation('elu')(z)

    elif activation_type == 'softplus':
        return Activation('softplus')(z)

    elif activation_type == 'minmax':
        return MinMax()(z)

    else:
        raise ValueError(f'unknown activation type: {activation_type}')

def safescad(input_shape=None,
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

def safescad2(input_shape=None,
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

def safescad_original(input_shape=None,
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


def cnn_2C2F(
        input_shape,
        num_classes,
        pooling='conv',
        activation='relu',
        initialization='orthogonal'):
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Conv2D(
        16, 4, strides=2, padding='same', kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)

    z = Conv2D(
        32, 4, strides=2, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)

    z = Flatten()(z)
    z = Dense(100, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y


def minmax_cnn_2C2F(
        input_shape,
        num_classes,
        pooling='conv',
        initialization='orthogonal'):
    return cnn_2C2F(
        input_shape, num_classes,
        pooling=pooling,
        activation='minmax',
        initialization=initialization)


def cnn_4C3F(
        input_shape,
        num_classes,
        pooling='conv',
        activation='relu',
        initialization='orthogonal'):
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Dense(512, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y


def minmax_cnn_4C3F(
        input_shape,
        num_classes,
        pooling='conv',
        initialization='orthogonal'):
    return cnn_4C3F(
        input_shape, num_classes,
        pooling=pooling,
        activation='minmax',
        initialization=initialization)


def cnn_6C2F(
        input_shape,
        num_classes,
        pooling='conv',
        activation='relu',
        initialization='orthogonal'):
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = Conv2D(32, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Conv2D(64, 3, padding='same', kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = _add_pool(z, pooling, activation, initializer)

    z = Flatten()(z)
    z = Dense(512, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y


def minmax_cnn_6C2F(
        input_shape,
        num_classes,
        pooling='conv',
        initialization='orthogonal'):
    return cnn_6C2F(
        input_shape, num_classes,
        pooling=pooling,
        activation='minmax',
        initialization=initialization)

def dense_small_3F(
        input_shape,
        num_classes,
        pooling='conv',
        activation='relu',
        initialization='orthogonal'):
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Dense(64, kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = Dense(32, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Dense(16, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y

def dense_med_3F(
        input_shape,
        num_classes,
        pooling='conv',
        activation='relu',
        initialization='orthogonal'):
    initializer = _get_initializer(initialization)

    x = Input(input_shape)
    z = Dense(128, kernel_initializer=initializer)(x)
    z = _add_activation(z, activation)
    z = Dense(64, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    z = Dense(32, kernel_initializer=initializer)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initializer)(z)

    return x, y