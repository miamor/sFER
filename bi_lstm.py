from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Activation, Flatten, Conv1D, TimeDistributed, Add

from rnn_utils import *
from util import *

import tflearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os


input_length = 512
frames = 80
# frames = 40
batch_size = 8
# batch_size = 64
num_classes = 2
epochs = 15
# epochs = 28
# epochs = 8
lstm_units = 10

'''
86.21%:
frames = 80
batch_size = 20
epochs = 20
dropout 0.8
'''

'''
def create_model():
    model = Sequential()
    # model.add(Embedding(None, 128, input_length=input_length))
    # model.add(Bidirectional(LSTM(128, input_shape=(frames, input_length), return_sequences=True)))
    # model.add(TimeDistributed(Conv1D(32, ())))
    model.add(Bidirectional(LSTM(64, input_shape=(
        frames, input_length), return_sequences=True)))
    # model.add(LSTM(64, input_shape=(frames, input_length), return_sequences=True))
    # model.add(Dropout(0.5))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model
'''


def create_model(mode=1):
    if mode == 1:
        input_layer = Input(shape=(frames, input_length, ))
        lstm = Bidirectional(
            LSTM(units=lstm_units, return_sequences=True))(input_layer)
        lstm = Bidirectional(
            LSTM(units=lstm_units, return_sequences=False))(lstm)

        dropout = Dropout(0.5)(lstm)
        # dropout = Flatten()(dropout)
        out = Dense(num_classes, activation='softmax')(dropout)

        model = Model(inputs=input_layer, outputs=out)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    elif mode == 2:
        input_layer = Input(shape=(frames, input_length, ))
        lstm_1 = Bidirectional(
            LSTM(units=lstm_units, return_sequences=False))(input_layer)

        input_layer__pair = Input(shape=(frames, input_length, ))
        lstm_1__pair = Bidirectional(
            LSTM(units=lstm_units, return_sequences=False))(input_layer__pair)

        merged_lstm = Add()([lstm_1, lstm_1__pair])

        dropout = Dropout(0.8)(merged_lstm)
        #flatten = Flatten()(dropout)
        out = Dense(num_classes, activation='softmax')(dropout)

        model = Model(inputs=[input_layer, input_layer__pair], outputs=out)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


def train(data, mode=1):
    print('Train...')

    model = create_model(mode=mode)

    (x_train, x_val, y_train, y_val) = data
    # (x_train, x_val, x_pair_train, x_pair_val, y_train, y_val) = data

    # get test data for evaluating on test set
    x_test, y_test, _, _ = get_data(
        'data/AFEW/seq_cnn_features/Val', frames, num_classes, input_length)
    # x_test, y_test, _, _, x_pair_test = get_data(
    #     'data/labeled-video-Val.pkl', frames, num_classes, input_length)

    if mode == 1:
        input_train = x_train
        input_val = x_val
        input_test = x_test
    # elif mode == 2:
    #     input_train = [x_train, x_pair_train]
    #     input_val = [x_val, x_pair_val]
    #     input_test = [x_test, x_pair_test]

    print(x_train.shape)
    print(y_train.shape)

    # hist=model.fit(x_train, y_train,
    #                  batch_size=batch_size,
    #                  epochs=epochs,
    #                  validation_data=[x_val, y_val],
    #                  verbose=1)
    hist = model.fit(input_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     validation_data=[input_val, y_val],
                     verbose=1)

    # Evaluate on validation set
    print("Evaluate on validation set")
    #score = model.evaluate(x_val, y_val, verbose=1)
    score = model.evaluate(input_val, y_val, verbose=1)
    print("%s: %.2f%%\n" % (model.metrics_names[1], score[1]*100))

    # Evaluate on test set
    print("Evaluate on test set")
    scoreTest = model.evaluate(input_test, y_test, verbose=1)

    print("%s: %.2f%%\n" % (model.metrics_names[1], scoreTest[1]*100))

    # Save model
    save_model_dir = 'model/new__' + \
        str(round(score[1]*100, 2))+'_'+str(round(scoreTest[1]*100, 2))
    if not os.path.isdir(save_model_dir):
        os.makedirs(save_model_dir)

    # # serialize model to JSON
    # save_model_name = save_model_dir+'/bi_lstm'
    # model_json = model.to_json()
    # with open(save_model_name+".json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights(save_model_name+"_weights.h5")

    # save all model and weights (current state) into 1 file
    save_model_name = save_model_dir+'/bi_lstm'
    model.save(save_model_name+'.h5')
    print("Saved model to disk")

    with open(save_model_name+"_history.json", 'w') as f:
        json.dump(hist.history, f)
        print("Saved training history to disk")

    return model, hist, save_model_name


def evaluate(data, model_name, mode=1):
    (X, Y, y, files) = data
    # (X, Y, y, files, X_pair) = data

    if mode == 1:
        input_test = X
    # elif mode == 2:
    #     [X, X_pair]

    # # load json and create model
    # json_file = open(model_name+'.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(model_name+"_weights.h5")

    # load & compile model
    loaded_model = load_model(model_name+'.h5')
    print("\nLoaded model from disk")
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])

    # Evaluate loaded model on test data
    score = loaded_model.evaluate(input_test, Y, verbose=1)
    print("Evaluate on test set")
    print("%s: %.2f%%\n" % (loaded_model.metrics_names[1], score[1]*100))

    # loaded_model.predict_class(X)
    y_pred = loaded_model.predict(X)
    predicted = np.argmax(y_pred, axis=1)

    print(predicted)
    print(np.array(y, dtype=int))

    # Stat
    total_neg = 0
    total_pos = 0
    false_neg = 0
    false_pos = 0
    for i, pred in enumerate(predicted):
        if y[i] == 0:
            total_neg += 1
        else:
            total_pos += 1

        if pred != y[i]:
            print('{}: {} ({})'.format(files[i], pred, y[i]))
            if pred == 1:
                false_pos += 1
            else:
                false_neg += 1
    print('Total positives: '+str(total_pos))
    print('Total negatives: '+str(total_neg))
    print('False positives: {}/{}={}'.format(false_pos,
                                             total_pos, 'null' if total_pos == 0 else false_pos/total_pos))
    print('False negatives: {}/{}={}'.format(false_neg,
                                             total_neg, 'null' if total_neg == 0 else false_neg/total_neg))
    # incorrects = np.nonzero(predicted != Y)
    # incorrects = np.nonzero(loaded_model.predict_class(X).reshape((-1,)) != Y)
    # print(incorrects)


def predict(x, model_name):
    # # load json and create model
    # json_file = open(model_name+'.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights(model_name+".h5")

    # load model
    loaded_model = load_model(model_name+'.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])

    preds = loaded_model.predict(x, verbose=0)
    np.set_printoptions(suppress=True)

    return preds


def plot_history(history):
    loss_list = [s for s in history.history.keys(
    ) if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys()
                     if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys(
    ) if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys()
                    if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(
            str(format(history.history[l][-1], '.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(
            str(format(history.history[l][-1], '.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(
            format(history.history[l][-1], '.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(
            format(history.history[l][-1], '.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    # data_train = get_training_data(
    #     'data/AFEW/seq_cnn_features/Train', frames, num_classes, input_length)

    # model, hist, model_name = train(data=data_train)
    # plot_history(hist)

    model_name = "model/new__60.0_92.31/bi_lstm"

    # data_test = get_data(
    #     'data/AFEW/seq_cnn_features/Val', frames, num_classes, input_length)
    # evaluate(data_test, model_name)

    # video_nums = ['000451280', '002607280', '000215738', '000257240', '000506080', '001343200', '001656240']
    # labels = [1, 1, 0,   1, 0, 0, 0]

    # video_nums = ['subject_020_spontaneous_smile_4_640x360_30', 'subject_020_posed_smile_2_640x360_30']
    # labels = [1, 0]

    # video_nums = ['S010__006', 'S026__006', 'S055__005']
    # labels = [0, 0, 0]

    # print(video_nums)
    # print(labels)
    # inputX = get_video_seq(frames, video_nums, labels)
    # predict(inputX, model_name)

    X_test, Y_test, _, _ = get_data('data/UvA/seq_cnn_features/Train', frames, num_classes, input_length)
    preds = predict(X_test, model_name)
    print(preds)
    print(Y_test)