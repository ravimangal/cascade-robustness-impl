import os, sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # with GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # with CPU
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

sys.path.append("..")
from netcal.scaling import TemperatureScaling
from gloro.models import GloroNet
from time import time
from scipy.special import softmax

class Confidence:
    def __init__(self, temperature_model, threshold):
        # calibrated_data = pd.read_csv(f'{data_dir}/test_calibrated.csv', header=None)
        # self.temperature_model = TemperatureScaling()
        # self.temperature_model.load_model(f'{model_dir}/temperature_model')
        self.temperature_model = temperature_model
        self.threshold = threshold

    def isVerified(self, x, pred):
        y_calibrated = self.temperature_model.transform(pred)
        pred_label = np.argmax(pred, axis=1)

        # In the case of binary classification, transform() only returns the probability
        # of the second label. Hence, we have to make the following modifications
        if pred.shape[1] == 2:
            y_calibrated_1 = np.ones(y_calibrated.shape, dtype=y_calibrated.dtype)
            y_calibrated_1 = y_calibrated_1 - y_calibrated
            y_calibrated = np.concatenate((np.expand_dims(y_calibrated_1,axis=1),
                                           np.expand_dims(y_calibrated,axis=1)), axis=1)
        ver_outcome = y_calibrated[np.arange(len(pred_label)), pred_label] >= self.threshold
        return ver_outcome

class GloRo:
    def __init__(self, model, epsilon):
        self.g = GloroNet(model=model, epsilon=epsilon)
        self.epsilon = epsilon

    def isVerified(self, x, pred):
        pred_label, eps, _ = self.g.predict_with_certified_radius(x)
        ver_outcome = eps >= self.epsilon
        return ver_outcome.numpy()


class VerifiedPredict:
    def __init__(self, model, temperature_model, epsilon, conf_threshold):
        self.model = model
        self.temperature_model = temperature_model
        self.v0 = GloRo(model, epsilon)
        self.v1 = Confidence(temperature_model, conf_threshold)

    # X_test is a numpy array of dim [batch_size, num_features] where batch_size can be 1
    def make_verified_prediction(self, X_test):
        y_pred = softmax(self.model.predict(X_test), axis=-1)
        v_res0 = self.v0.isVerified(X_test, y_pred)
        v_res1 = self.v1.isVerified(X_test, y_pred)
        pred_label = np.argmax(y_pred, axis=1)
        return (pred_label, v_res0, v_res1)



