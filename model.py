import numpy as np
import random

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Permute, Conv1D, BatchNormalization, GlobalAveragePooling1D, \
    concatenate
from keras.layers import LSTM
from keras import optimizers, regularizers
from keras.layers import Bidirectional
from keras import backend as K


class model:

    def __init__(self, opt="adam"):

        if opt == "adam":
            self.opt = optimizers.adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        elif opt == "rmsprop":
            self.opt = optimizers.rmsprop()
        elif opt == "sgd":
            self.opt = optimizers.SGD(clipvalue=0.5)

        else:
            raise Exception('Enter a valid optimizer')

    def f1(self, y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def single_class_accuracy(self, interesting_class_id):
        def fn(y_true, y_pred):
            class_id_preds = K.argmax(y_pred, axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            positive_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
            true_mask = K.cast(K.equal(y_true, interesting_class_id), 'int32')
            acc_mask = K.cast(K.equal(positive_mask, true_mask), 'float32')
            class_acc = K.mean(acc_mask)
            return class_acc

        return fn

    def create_lstm(self, X, y):

        ml_model = Sequential()
        ml_model.add(LSTM(32, return_sequences=False, input_shape=(X.shape[1], X.shape[-1])))
        ml_model.add(Dropout(0.5))
        ml_model.add(Dense(y.shape[-1], activation='softmax'))

        ml_model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=[self.f1, "accuracy"])

        return ml_model

    def create_blstm(self, X, y):

        ml_model = Sequential()

        ml_model.add(Bidirectional(LSTM(8, return_sequences=False), input_shape=(X.shape[1], X.shape[-1])))
        ml_model.add(Dense(y.shape[-1], activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l2(0.01)))
        ml_model.compile(loss='categorical_crossentropy', optimizer=self.opt, metrics=eval_metric)

        return ml_model