import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scriptify import scriptify
import architectures
from utils import setup_nnet_tools, compute_nnet_params, save_nnet
from data_preprocessing import get_data
from gloro.models import GloroNet
from gloro.training import losses
from gloro.training.callbacks import EpsilonScheduler
from gloro.training.callbacks import LrScheduler
from gloro.training.metrics import clean_acc
from gloro.training.metrics import vra
from gloro.training.metrics import rejection_rate

if __name__ == '__main__':

    @scriptify
    def script(experiment='safescad',
               arch='safescad',
               epochs=30,
               batch_size=128,
               dataset_file=None,
               conf_name='default',
               epsilon=0.5,
               lr=0.01,
               gpu=0):
        print("Configuration Options:")
        print("experiment=", experiment)
        print("arch=", experiment)
        print("epochs=", epochs)
        print("batch_size=", batch_size)
        print("dataset_file=", dataset_file)
        print("conf_name=", conf_name)
        print("epsilon=", epsilon)
        print("lr=", lr)


        # Select the GPU and allow memory growth to avoid taking all the RAM.
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        #Basic setup and install additional dependencies
        # Some global variables and general settings
        model_dir = f'./experiments/models/{experiment}/{conf_name}'
        data_dir = f'./experiments/data/{experiment}/{conf_name}'
        pd.options.display.float_format = '{:.2f}'.format
        nnet_tools_path = os.path.abspath('NNet')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # setup nnet tools (for converting model to Stanford's nnet format)
        setup_nnet_tools(nnet_tools_path)

        # Load and Preprocess Dataset
        X_train_enc, y_train_enc, X_test_enc, y_test_enc = get_data(experiment, dataset_file, data_dir)
        print("# of train samples: ", np.shape(X_train_enc)[0])
        print("# of test samples: ", np.shape(X_test_enc)[0])
        ## Build & Train NN
        n_categories = y_train_enc.shape[1]
        arch = getattr(architectures, f'{arch}')
        model = arch((X_train_enc.shape[1],), classes=n_categories)

        gloro_model = GloroNet(model=model, epsilon=epsilon, num_iterations=5)
        gloro_model.summary()
        gloro_model.compile(
            loss=losses.get('crossentropy'),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            metrics=[clean_acc, vra, rejection_rate])

        history = gloro_model.fit(X_train_enc, y_train_enc,
                                  validation_data=(X_test_enc, y_test_enc),
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  callbacks=[EpsilonScheduler('fixed'),
                                             LrScheduler('decay_to_0.0001'),])

        # evaluate the model
        print('Evaluating model ...')
        train_eval = gloro_model.evaluate(X_train_enc, y_train_enc)
        test_eval = gloro_model.evaluate(X_test_enc, y_test_enc)
        results = {}
        results.update({
            'test_' + metric.name.split('pred_')[-1]: round(value, 4)
            for metric, value in zip(gloro_model.metrics, test_eval)
        })
        results.update({
            'train_' + metric.name.split('pred_')[-1]: round(value, 4)
            for metric, value in zip(gloro_model.metrics, train_eval)
        })
        print(results)

        model = gloro_model.f

        # save model in tf and h5 formats
        tf_model_path = f'{model_dir}/model.tf'
        h5_model_path = f'{model_dir}/model.h5'
        model.save(tf_model_path)  # save_format='tf'
        model.save(h5_model_path, save_format='h5')


        # extract params for nnet format
        nnet_params = compute_nnet_params(tf_model_path, np.concatenate((X_train_enc,X_test_enc)))
        weights, biases, input_mins, input_maxs, means, ranges = nnet_params

        # write the model to nnet file. Note that the final softmax operation is not recorded in the nnet file.
        # This does not affect any of the verification queries we pose to Marabou.
        nnet_path = os.path.join(model_dir, f'model.nnet')
        save_nnet(weights, biases, input_mins, input_maxs, means, ranges, nnet_path)
