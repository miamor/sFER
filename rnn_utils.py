"""
Utilities used by our other RNN scripts.
"""
from collections import deque
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
import tflearn
import numpy as np
import pickle
import os


def pad(inputX, frames):
    i = len(inputX)
    # print(inputX)
    leftframes = frames - i
    if leftframes > 0:
        zeros = np.zeros([leftframes, 512])
        # print(zeros)
        inputX = np.concatenate((inputX, zeros), axis=0)
        # inputX = np.array()
        # print(inputX)
    return inputX


def get_data(listfile, num_frames, num_classes, input_length):
    X = []
    y = []
    # temp_list = deque()

    with open(listfile, 'rb') as fin:
        labeled_videos = pickle.load(fin)  # simply [[video, label],...]
        for video_label in labeled_videos:
            video = video_label[0]
            label = video_label[1]

            temp_list = []

            ft_file = 'data/cnn_features/cnn-features-video-' + video + '.pkl'
            with open(ft_file, 'rb') as ftin:
                features = pickle.load(ftin)

                i = 0
                for ft in features:
                    i = i+1
                    if i <= num_frames:
                        temp_list.append(ft)

            flat = np.array(list(temp_list))

            if i < num_frames:
                zeros = np.zeros((num_frames-i, 512))
                flat = np.concatenate((flat, zeros), axis=0)

            X.append(flat)
            y.append(label)

    X = np.array(X)
    y = to_categorical(y, num_classes)

    return X, y


def get_training_data(listfile, num_frames, num_classes, input_length):
    X, y = get_data(listfile, num_frames, num_classes, input_length)

    # Split into train and test.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42)

    return X_train, X_val, y_train, y_val


def get_video_seq(num_frames, video_nums, labels):
    """Get the data from our saved predictions or pooled features."""

    X = []

    # for video in video_nums:
    l = 0
    while l < len(video_nums):
        video = video_nums[l]
        #label = labels[l]
        l += 1

        temp_list = []

        ft_file = 'data/cnn_features/cnn-features-video-' + video + '.pkl'
        with open(ft_file, 'rb') as ftin:
            features = pickle.load(ftin)

            i = 0
            for ft in features:
                i = i+1
                if i <= num_frames:
                    temp_list.append(ft)

        flat = np.array(list(temp_list))

        if i < num_frames:
            zeros = np.zeros((num_frames-i, 512))
            flat = np.concatenate((flat, zeros), axis=0)

        X.append(flat)

        #X = np.concatenate(X, flat)

    X = np.array(X)

    return X


def get_network(frames, input_size, num_classes):
    """Create our LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 128, dropout=0.8, return_seq=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def get_network_sm(frames, input_size, num_classes):
    """Create our LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 64, dropout=0.8, return_seq=True)
    net = tflearn.lstm(net, 64)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='binary_crossentropy', name="output1")
    return net


def get_network_deep(frames, input_size, num_classes):
    """Create a deeper LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def get_network_wide(frames, input_size, num_classes):
    """Create a wider LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 256, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net


def get_network_wider(frames, input_size, num_classes):
    """Create a wider LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 512, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net


def get_network_wider_deep(frames, input_size, num_classes):
    """Create a wider LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    # net = tflearn.lstm(net, 512, dropout=0.2)
    net = tflearn.lstm(net, 512, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 512, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 512, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net
