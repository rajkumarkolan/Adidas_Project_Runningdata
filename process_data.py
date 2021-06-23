import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import re

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, Permute, Conv1D, BatchNormalization, GlobalAveragePooling1D, \
    concatenate
from keras.layers import LSTM
from keras import optimizers, regularizers
from keras.layers import Bidirectional

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import np_utils


class process_data:

    #    def __init__(self):

    def create_time_window(self, data, time_window=250, overlapping=False):

        dataset = np.zeros((0, time_window, data.shape[-1] - 1))
        y = np.zeros((0))

        for k in range(data['terrain'].nunique()):
            windows = (data[data['terrain'] == k].shape[0] - data[data['terrain'] == k].shape[0] % time_window)

            if overlapping:
                overlapping_windows = np.linspace(0, windows, 2 * (windows / time_window) + 1)[:-2]
                tr_dataset = np.zeros((len(overlapping_windows), time_window, data.shape[-1] - 1))

            else:
                overlapping_windows = np.linspace(0, windows, (windows / time_window) + 1)[:-2]
                tr_dataset = np.zeros((len(overlapping_windows), time_window, data.shape[-1] - 1))

            for index, start in enumerate(overlapping_windows):
                end = start + time_window
                tr_data = data[data['terrain'] == k].drop(['terrain'], axis=1).values[int(start):int(end)]
                tr_dataset[index, :, :] = tr_data.reshape(1, time_window, tr_data.shape[-1])

            y = np.concatenate((y, k * np.ones((tr_dataset.shape[0]))))
            dataset = np.concatenate((dataset, tr_dataset), axis=0)

        lb = LabelBinarizer()
        lb.fit(y)
        y = lb.transform(y)

        return dataset, y

    def process_subjects(self, data, subjects=[19]):

        sub_len = []

        dat = np.zeros((1, 14))

        for i in data.keys():
            #       subjects = [19]
            if int(re.findall('\d+', str(i))[0]) in subjects:
                print("parsing data for ", int(re.findall('\d+', str(i))[0]))
                group = data[i]  # Names of the groups in HDF5 file.

                for j in group.keys():
                    if j == 'subject_meta_info':
                        break
                    member = group[j]

                    for k in member.keys():
                        if k == 'Left':
                            foot = 0 * np.ones((np.array(member[k]).shape[0], 1))
                            subject = int(re.findall('\d+', str(i))[0]) * np.ones((np.array(member[k]).shape[0], 1))
                            temp = np.concatenate((np.array(member[k]), foot, subject), axis=1)
                            dat = np.concatenate((dat, temp), axis=0)
                        elif k == 'Right':
                            foot = 1 * np.ones((np.array(member[k]).shape[0], 1))
                            subject = int(re.findall('\d+', str(i))[0]) * np.ones((np.array(member[k]).shape[0], 1))
                            temp = np.concatenate((np.array(member[k]), foot, subject), axis=1)
                            dat = np.concatenate((dat, temp), axis=0)

        df = pd.DataFrame(dat,
                          columns=['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'magX', 'magY', 'magZ', 'terrain',
                                   'pain', 'exhaustion', 'foot', 'subject'])

        df = df.drop(df.index[0])

        return df

    def normalize_data(self, data):
        for col in data.columns[:-1]:
            data[col] = (data[col] - np.min(data[col])) / (np.max(data[col]) - np.min(data[col]))

        return data

    def split_data(self, data, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=42,shuffle=True)

        return X_train, X_test, y_train, y_test

    def split_balanced(self, data, target, test_size=0.2):

        classes = np.unique(target)
        # can give test_size as fraction of input data size of number of samples
        if test_size < 1:
            n_test = np.round(len(target) * test_size)
        else:
            n_test = test_size
        n_train = max(0, len(target) - n_test)
        n_train_per_class = max(1, int(np.floor(n_train / len(classes))))
        n_test_per_class = max(1, int(np.floor(n_test / len(classes))))

        ixs = []
        for cl in classes:
            if (n_train_per_class + n_test_per_class) > np.sum(target == cl):
                # if data has too few samples for this class, do upsampling
                # split the data to training and testing before sampling so data points won't be
                #  shared among training and test data
                splitix = int(
                    np.ceil(n_train_per_class / (n_train_per_class + n_test_per_class) * np.sum(target == cl)))
                ixs.append(np.r_[np.random.choice(np.nonzero(target == cl)[0][:splitix], n_train_per_class),
                                 np.random.choice(np.nonzero(target == cl)[0][splitix:], n_test_per_class)])
            else:
                ixs.append(np.random.choice(np.nonzero(target == cl)[0], n_train_per_class + n_test_per_class,
                                            replace=False))

        # take same num of samples from all classes
        ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
        ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class + n_test_per_class)] for x in ixs])

        X_train = data[ix_train, :]
        X_test = data[ix_test, :]
        y_train = target[ix_train]
        y_test = target[ix_test]

        return X_train, X_test, y_train, y_test
