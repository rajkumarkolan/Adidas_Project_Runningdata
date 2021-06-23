import sys
import h5py
import numpy as np
import pandas as pd

import process_data
import model

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

if __name__ == '__main__':
    print("Extracting the data")
    # print("\n")
    data = h5py.File("./Dataset/running_data_set.h5", 'r')

    # Select the subjects depending on the amount of data you need to train the model
    # Initially, I'm selecting only subject 19 because I'm thinking of building a simple small model
    # which predicts the labels for 1 user. My intuition tells me, as we use data from different
    # users, we might confuse the model as ever user has their own running pattern and pace.
    subjects = [16, 17, 18, 19, 20, 21]
    # subjects = [19]

    # the function extracts the data under the mentioned subjects and stores them in a pandas dataframe
    # with 13 features and label as 'terrain'
    print("Parsing the data for " + str(subjects))
    # print("\n")
    process_dataset = process_data.process_data()
    df = process_dataset.process_subjects(data, subjects)

    print("data shape: ", df.shape)

    print("Selecting only the left foot")
    # print("\n")

    # I consider only the left foot for now!
    df = df[df['foot'] == 0]

    print("data shape: ", df.shape)

    # I'm dropping the features so that the classification can be done based on sequences.
    data = df.drop(['foot', 'pain', 'exhaustion', 'subject'], axis=1)

    # Normalizing the dataset
    # data = process_dataset.normalize_data(data)

    data["terrain"] = data["terrain"] - 1

    # Grouping similar labels
    # data = data[~data['terrain'].isin([1,2,3,4])]
    new_data = data.copy()

    new_data["terrain"][data.terrain == 2] = 1
    new_data["terrain"][data.terrain == 1] = 2
    new_data["terrain"][data.terrain == 3] = 2
    new_data["terrain"][data.terrain == 4] = 3
    new_data["terrain"][data.terrain == 5] = 3

    # Dropping unused features
    new_data = new_data.drop(["accX", "accY", "gyroX", "gyroY", "gyroZ", "magX", "magY", "magZ"], axis=1)

    # 1. Creating windows and corresponding label with constant window size

    # Creating windows in the dataset (sequences of constant length with 1 label for each window)
    # The overlapping parameter tells the function to create a rolling window or not

    time_window = 800

    flag = False
    dataset, y = process_dataset.create_time_window(data=new_data, time_window=time_window, overlapping=flag)

    # 2. Creating window based on the runner stance (mid-stance segmentation) I'm just seeing if i
    # create windows based on mid stances, if my model learns patterns more consistently. #will be added later

    # split data
    X_train, X_test, y_train, y_test = process_dataset.split_balanced(data=dataset, target=y, test_size=0.2)

    print("Number of train datapoints: ", X_train.shape[0])
    print("Number of test datapoints: ", X_test.shape[0])
    print("Window size: ", time_window)
    print("Window Overlapping: " , str(flag))
    print("Number of features: ", X_train.shape[-1])

    # shuffle data

    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)

    # create model
    Model = model.model(opt="rmsprop")

    # lstm
    ml_model = Model.create_lstm(X_train, y_train)

    print(ml_model.summary())

    ml_model.fit(X_train, y_train, batch_size=32, epochs=200)
    score = ml_model.evaluate(X_test, y_test, batch_size=32)

    print("Test Accuracy: %.2f%%" % (score[2] * 100))
    print("Test F1_score: ", score[1])

  
