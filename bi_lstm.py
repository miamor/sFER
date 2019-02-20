from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Activation, Flatten
from keras.models import model_from_json

from rnn_utils import get_network, get_network_deep, get_network_wide, get_network_wider, get_network_wider_deep, get_data, get_network_sm, pad, get_video_seq
import tflearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

input_length = 512
#frames = 80
frames = 40
#batch_size = 128
batch_size = 64
num_classes = 2
#epochs = 30
epochs = 10
#epochs = 8
model_name = "bi_lstm"

'''
86.21%:
frames = 80
batch_size = 20
epochs = 20
dropout 0.8
'''


def create_model_():
    model = Sequential()
    #model.add(Embedding(None, 128, input_length=input_length))
    #model.add(Bidirectional(LSTM(128, input_shape=(80,512), return_sequences=True)))
    model.add(LSTM(64, input_shape=(frames, input_length), return_sequences=True))
    model.add(Dropout(0.5))
    #model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    return model

def create_model(model_name):
    inputs = Input(shape=(None,512))
    #x = Bidirectional(LSTM(128, return_sequences=False, kernel_initializer='Orthogonal'))(x)
    x = Bidirectional(LSTM(128, return_sequences=False))(inputs)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='softmax')(x)
    #x = Activation('elu')(x)

    model = Model(inputs, x, name=model_name)  # load weights

    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return model

def train(model, data, model_name):
    print('Train...')
    (x_train, x_test, y_train, y_test) = data

    print(x_train.shape)
    print(y_train.shape)

    hist = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_test, y_test],
              verbose=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model/"+model_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/"+model_name+".h5")
    print("Saved model to disk")

    with open("model/"+model_name+"_history.json", 'w') as f:
        json.dump(hist.history, f)
        print("Saved training history to disk")

    return model, hist


def evaluate(data, model_name):
    (_, X, _, Y) = data

    # load json and create model
    json_file = open('model/'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/"+model_name+".h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

    ''' plot loss 
    # list all data in loaded_model
    print(loaded_model.history.keys())
    # summarize loaded_model for accuracy
    plt.plot(loaded_model.history['acc'])
    plt.plot(loaded_model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize loaded_model for loss
    plt.plot(loaded_model.history['loss'])
    plt.plot(loaded_model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''

def predict(x, model_name):
    # load json and create model
    json_file = open('model/'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/"+model_name+".h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop', metrics=['accuracy'])
    
    preds = loaded_model.predict(x, verbose=1)
    np.set_printoptions(suppress=True)
    print(preds)


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def load_data():
    # Get our data.
    X_train_, X_test_, y_train, y_test = get_data(
        frames, num_classes, input_length)

    X_train = []
    X_test = []
    for x_train in X_train_:
        X_train.append(pad(x_train, frames))
    for x_test in X_test_:
        X_test.append(pad(x_test, frames))

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    print(len(X_train))
    print(len(X_train[0]))
    print(len(X_train[0][0]))

    print("X_train[0]: {}".format(X_train[0]))

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data = load_data()

    #print(X_train)

    model = create_model_()
    #model = create_model(model_name="bi_lstm")
    model, hist = train(model=model, model_name=model_name, data=data)
    print(model.summary())
    evaluate(data, model_name)
    plot_history(hist)

    #inputX = np.array([data[1][0]])
    video_nums = ['000257240', '000506080', '001343200', '001656240']
    labels = [1, 0, 0, 0]
    print(video_nums)
    print(labels)
    inputX = get_video_seq(frames, video_nums, labels)
    print(inputX.shape)
    predict(inputX, model_name)