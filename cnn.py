from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import model_from_json

import tflearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import cv2

batch_size = 16
num_classes = 2
epochs = 10
(img_w, img_h) = (64, 64)


def resize_img(img_path):
    oriimg = cv2.imread(img_path)
    '''
    height, width, depth = oriimg.shape
    imgScale = W/width
    newX, newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
    newimg = cv2.resize(oriimg, (int(newX), int(newY)))
    '''
    newimg = cv2.resize(oriimg, (64, 64))

    file_name = img_path.split('/')[-1]
    save_dir = img_path.split(file_name)[0].replace(
        '_half', '_half_resized')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    new_img_path = os.path.join(save_dir, file_name)

    if not os.path.exists(new_img_path):
        cv2.imwrite(new_img_path, newimg)


def resize_imgs(dirname):
    dDir = '/media/tunguyen/Devs/DeepLearning/FacialExpression/keras-vggface/data/CK+/extracted/'+dirname
    for folder in os.listdir(dDir):
        folder = os.path.join(dDir, folder)
        for cf in os.listdir(folder):
            # if 'S057' in folder and cf == '006':
            if folder != '0':
                cf = os.path.join(folder, cf)
                print(cf)
                for f in glob.glob(os.path.join(cf, "*.png")):
                    img_path = os.path.join(cf, f)
                    resize_img(img_path)


def save_data_npy(dirname):
    dDir = '/media/tunguyen/Devs/DeepLearning/FacialExpression/keras-vggface/data/CK+/extracted/'+dirname+'_resized'

    X = []
    Y = []
    total_img = 0
    for folder in os.listdir(dDir):
        folder = os.path.join(dDir, folder)
        for cf in os.listdir(folder):
            # if 'S057' in folder and cf == '006':
            if folder != '0':
                cf = os.path.join(folder, cf)
                for f in glob.glob(os.path.join(cf, "*.png")):
                    img_path = os.path.join(cf, f)
                    #print(img_path)
                    
                    img = cv2.imread(img_path, 0)

                    X.append(img)
                    Y.append(1)

                    total_img += 1

    X = np.array(X)
    print(X.shape)
    print(total_img)

    '''
    x_train, x_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.1, random_state=42)
    np.save('data/x_train.npy', x_train)
    np.save('data/x_val.npy', x_val)
    '''
    np.save('data/x_'+dirname+'.npy', X)
    
    return X


def load_data(dirname):
    with open('data/x_'+dirname+'.npy', 'rb') as fin:
        X = np.load(fin)


def upper_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, padding=1, activation='relu',
                     input_shape=(img_w, img_h, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, padding=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(128, kernel_size=3, padding=0, strides=2, activation='relu'))
    model.add(Conv2D(256, kernel_size=3, padding=1, strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, padding=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model


def lower_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, padding=1, activation='relu',
                     input_shape=(img_w, img_h, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, padding=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, kernel_size=3, padding=0, strides=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(256, kernel_size=3, padding=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])


def train(model, data):
    print('Train...')
    (x_train, x_val, y_train, y_val) = data

    print(x_train.shape)
    print(y_train.shape)

    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=[x_val, y_val],
                     verbose=1)

    print("Evaluate on validation set")
    score = model.evaluate(x_val, y_val, verbose=1)
    print("%s: %.2f%%\n" % (model.metrics_names[1], score[1]*100))

    return model, hist



if __name__ == '__main__':
    #resize_imgs('upper_half')

    #save_data_npy('lower_half')