import re
import gc

import keras.layers as layers
from keras.layers import Bidirectional
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from numpy.random import seed

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import numpy as np
from time import gmtime, strftime

import os
from os import listdir
from os.path import isfile, join, isdir

import pickle

"""# EDA helpers"""


def get_x_y(train_txt, num_classes, word2vec_len, input_size, word2vec, percent_dataset):
    train_lines = open(train_txt, 'r').readlines()
    shuffle(train_lines)
    train_lines = train_lines[:int(percent_dataset * len(train_lines))]
    num_lines = len(train_lines)

    x_matrix = None
    y_matrix = None

    try:
        x_matrix = np.zeros((num_lines, input_size, word2vec_len))
    except:
        print("Error!", num_lines, input_size, word2vec_len)
    y_matrix = np.zeros((num_lines, num_classes))

    for i, line in enumerate(train_lines):
        parts = line[:-1].split('\t')
        label = int(parts[0])
        sentence = parts[1]

        words = sentence.split(' ')
        words = words[:x_matrix.shape[1]]
        for j, word in enumerate(words):
            if word in word2vec:
                x_matrix[i, j, :] = word2vec[word]
        
        y_matrix[i][label] = 1.0
        
    return x_matrix, y_matrix


def build_model(sentence_length, word2vec_len, num_classes):
    model = None
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(sentence_length, word2vec_len)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(32, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def one_hot_to_categorical(y):
    assert len(y.shape) == 2
    return np.argmax(y, axis=1)


def run_model(train_file, test_file, num_classes, percent_dataset):
    model = build_model(input_size, word2vec_len, num_classes)
    
    train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec, percent_dataset)
    test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec, 1)

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    print('Before fitting')
    model.fit(train_x,
              train_y,
              epochs=100000,
              callbacks=callbacks,
              validation_split=0.1,
              batch_size=1024,
              shuffle=True,
              verbose=1)
    print('After fitting')
    y_pred = model.predict(test_x)
    test_y_cat = one_hot_to_categorical(test_y)
    y_pred_cat = one_hot_to_categorical(y_pred)
    acc = accuracy_score(test_y_cat, y_pred_cat)

    train_x, train_y = None, None
    gc.collect()

    return acc


def load_pickle(file):
    return pickle.load(open(file, 'rb'))


def get_now_str():
    return str(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))

"""# Set variables

You need to mount your drive with the following shared folder
[Data folder](https://drive.google.com/drive/folders/1o-NLh282O24Q4aaPA31f90ahFcQGvfb1?usp=sharing)
"""

# https://drive.google.com/drive/folders/1o-NLh282O24Q4aaPA31f90ahFcQGvfb1?usp=sharing
root = '/home/tarik/codeyard/master/dap/seminarski'
datasets = ['pc']
dataset_folders = [root + '/increment_datasets_f2/' + dataset for dataset in datasets] 

num_classes_list = [2]

increments = [0.7, 0.8, 0.9, 1]

input_size_list = [25]

huge_word2vec = root + '/word2vec/glove.840B.300d.txt'
word2vec_len = 300

"""# Execute training and testing"""

orig_accs = {dataset: {} for dataset in datasets}
aug_accs = {dataset: {} for dataset in datasets}

writer = open(root + '/outputs_f2/' + get_now_str() + '.csv', 'w+')

for i, dataset_folder in enumerate(dataset_folders):

    dataset = datasets[i]
    num_classes = num_classes_list[i]
    input_size = input_size_list[i]
    train_orig = dataset_folder + '/train_orig.txt'
    train_aug_st = dataset_folder + '/train_aug_st.txt'
    test_path = dataset_folder + '/test.txt'
    word2vec_pickle = dataset_folder + '/word2vec.p'
    word2vec = load_pickle(word2vec_pickle)

    for increment in increments:
        aug_acc = run_model(train_aug_st, test_path, num_classes, increment)

        aug_accs[dataset][increment] = aug_acc

        orig_acc = run_model(train_orig, test_path, num_classes, increment)
        orig_accs[dataset][increment] = orig_acc

        print(dataset, increment, orig_acc, aug_acc)
        writer.write(dataset + ',' + str(increment) + ',' + str(orig_acc) + ',' + str(aug_acc) + '\n')

        gc.collect()

print(orig_accs, aug_accs)
