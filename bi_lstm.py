from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Activation, Flatten
from keras.models import model_from_json

from rnn_utils import get_network, get_network_deep, get_network_wide, get_network_wider, get_network_wider_deep, get_network_sm, pad, get_video_seq, get_data, get_training_data
import tflearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os

input_length = 512
#frames = 80
frames = 40
#batch_size = 128
batch_size = 16
num_classes = 2
#epochs = 15
epochs = 10
#epochs = 8

'''
86.21%:
frames = 80
batch_size = 20
epochs = 20
dropout 0.8
'''


def create_model():
    model = Sequential()
    #model.add(Embedding(None, 128, input_length=input_length))
    #model.add(Bidirectional(LSTM(128, input_shape=(80,512), return_sequences=True)))
    model.add(LSTM(64, input_shape=(frames, input_length), return_sequences=True))
    #model.add(Dropout(0.5))
    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model


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

    x_test, y_test = get_data(
        'data/labeled-video-Val.pkl', frames, num_classes, input_length)
    print("Evaluate on test set")
    scoreTest = model.evaluate(x_test, y_test, verbose=1)
    print("%s: %.2f%%\n" % (model.metrics_names[1], scoreTest[1]*100))

    save_model_dir = 'model/'+str(round(score[1]*100, 2))+'_'+str(round(scoreTest[1]*100, 2))
    if not os.path.isdir(save_model_dir):
        os.makedirs(save_model_dir)
        
    # serialize model to JSON
    save_model_name = save_model_dir+'/bi_lstm'
    model_json = model.to_json()
    with open(save_model_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_model_name+".h5")
    print("Saved model to disk")

    with open(save_model_name+"_history.json", 'w') as f:
        json.dump(hist.history, f)
        print("Saved training history to disk")

    return model, hist, save_model_name


def evaluate(data, model_name):
    (X, Y) = data

    # load json and create model
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+".h5")
    print("\nLoaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=1)
    print("Evaluate on test set")
    print("%s: %.2f%%\n" % (loaded_model.metrics_names[1], score[1]*100))



def predict(x, model_name):
    # load json and create model
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+".h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])

    preds = loaded_model.predict(x, verbose=0)
    np.set_printoptions(suppress=True)
    print(preds)


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
    #data = load_data()

    # print(X_train)

    data_train = get_training_data(
        'data/labeled-video-Train.pkl', frames, num_classes, input_length)

    model = create_model()
    model, hist, model_name = train(model=model, data=data_train)
    plot_history(hist)
    #print(model.summary())

    '''
    #model_name = "86.21_83.06/bi_lstm"
    data_test = get_data(
        'data/labeled-video-Val.pkl', frames, num_classes, input_length)
    evaluate(data_test, model_name)
    '''

    video_nums = ['000451280', '002607280', '000215738', '000257240', '000506080', '001343200', '001656240']
    labels = [1, 1, 0,   1, 0, 0, 0]
    print(video_nums)
    print(labels)
    inputX = get_video_seq(frames, video_nums, labels)
    predict(inputX, model_name)
