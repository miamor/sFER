from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, Activation, Flatten, Conv1D, TimeDistributed, Add, RepeatVector, Permute, Multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger

from rnn_utils import *
from util import *

import tflearn
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse


from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib


input_length = 512
frames = 50
# frames = 30
# batch_size = 32
batch_size = 128
num_classes = 2
epochs = 400
# epochs = 28
# epochs = 8
lstm_units = 7

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
        # lstm = Bidirectional(
        #     LSTM(units=lstm_units, return_sequences=False))(lstm)

        dropout = Dropout(0.8)(lstm)
        flatten = Flatten()(dropout)
        out = Dense(num_classes, activation='softmax')(flatten)

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
    elif mode == 3:
        input_layer = Input(shape=(frames, input_length, ))
        input_dims = 10

        # attention layer
        attention_probs = Dense(input_length, activation='softmax')(input_layer)
        attention_mul = Multiply()([input_layer, attention_probs])

        lstm = Bidirectional(
            LSTM(units=lstm_units, return_sequences=True))(attention_mul)
        # lstm = Bidirectional(
        #     LSTM(units=lstm_units, return_sequences=False))(lstm)

        # # compute importance for each step
        # attention = Dense(1, activation='tanh')(lstm)
        # attention = Flatten()(attention)
        # attention = Activation('softmax')(attention)
        # attention = RepeatVector(lstm_units)(attention)
        # attention = Permute([2, 1])(attention)

        # rep = Multiply()([lstm, attention])


        dropout = Dropout(0.8)(lstm)
        flatten = Flatten()(dropout)
        out = Dense(num_classes, activation='softmax')(flatten)

        model = Model(inputs=input_layer, outputs=out)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


def train(data, model_path=None, mode=1):
    print('Train...')


    # ## Load data
    (x_train, x_val, y_train, y_val) = data
    # (x_train, x_val, x_pair_train, x_pair_val, y_train, y_val) = data



    input_train = x_train
    input_val = x_val
    # input_test = x_test
    # if mode == 2:
    #     input_train = [x_train, x_pair_train]
    #     input_val = [x_val, x_pair_val]
    #     input_test = [x_test, x_pair_test]

    print(x_train.shape)
    print(y_train.shape)


    # ## Create or load model
    if model_path == None:
        model = create_model(mode=mode)
        save_dir = 'checkpoints/MAHNOB_frames_'+str(frames)+'_'+str(args.train_mode)
    else:
        model = load_model(model_path)
        save_dir = 'checkpoints/'+model_path.split('/')[-2]



    # ## Fit model
    model_checkpoint = ModelCheckpoint(filepath=save_dir+'/bi_lstm-{epoch:02d}_loss-{loss:.5f}_val_loss-{val_loss:.5f}.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto',
                                       period=1)
    # csv_logger = CSVLogger(filename=out_dir+'/vae_lstm_training_log.csv',
    #                       separator=',',
    #                       append=True)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0,
                                   patience=20,
                                   verbose=1)
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.2,
                                             patience=6,
                                             verbose=1,
                                             epsilon=0.001,
                                             cooldown=0,
                                             min_lr=0.00001)
    callbacks = [model_checkpoint,
                 # csv_logger,
                 early_stopping,
                 reduce_learning_rate
                 ]

    history = model.fit(input_train, y_train,
                      validation_data=(input_val, y_val),
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=callbacks,
                      verbose=1)





    # Save model
    save_model_dir = 'output/bi_lstm__MAHNOB_12h49/'
    if not os.path.isdir(save_model_dir):
        os.makedirs(save_model_dir)
    save_model_name = save_model_dir+'/bi_lstm'
    model.save(save_model_name+'.h5')
    print("Saved model to disk")


    # save_model_dir = 'output/new__' + \
    #     str(round(score[1]*100, 2))+'_'+str(round(scoreTest[1]*100, 2))
    # if not os.path.isdir(save_model_dir):
    #     os.makedirs(save_model_dir)

    # # serialize model to JSON
    # save_model_name = save_model_dir+'/bi_lstm'
    # model_json = model.to_json()
    # with open(save_model_name+".json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights(save_model_name+"_weights.h5")

    # # save all model and weights (current state) into 1 file
    # save_model_name = save_model_dir+'/bi_lstm'
    # model.save(save_model_name+'.h5')
    # print("Saved model to disk")

    # with open(save_model_name+"_history.json", 'w') as f:
    #     json.dump(hist.history, f)
    #     print("Saved training history to disk")

    return model, history, save_model_name


def evaluate(data, model_path, mode=1):
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
    # loaded_model.compile(loss='binary_crossentropy',
    #                      optimizer='rmsprop', metrics=['accuracy'])
    # # load weights into new model
    # loaded_model.load_weights(model_name+".h5")

    # load & compile model
    loaded_model = load_model(model_path)
    print("\nLoaded model from disk")
    loaded_model.summary()

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
            # print('{}: {} ({})'.format(files[i], pred, y[i]))
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




def train_classifier(X_train, Y_train, clf='CART', save_path=None):
    """
    Spot Check Algorithms

    This is just for inspecting which classifier could possibly work best with our data
    """
    # models = []
    # models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    # models.append(('SVM', SVC()))
    # # evaluate each model in turn
    # results = []
    # names = []
    # for name, model in models:
    #     kfold = model_selection.KFold(n_splits=10, random_state=7)
    #     cv_results = model_selection.cross_val_score(
    #         model, X_train, Y_train, cv=kfold, scoring='accuracy')
    #     results.append(cv_results)
    #     names.append(name)
    #     print("{}: {} ({})".format(name, cv_results.mean(), cv_results.std()))

    """
    Train the classifier
    """
    if clf == 'CART':
        classifier = DecisionTreeClassifier()
    elif clf == 'LDA':
        classifier = LinearDiscriminantAnalysis()
    elif clf == 'LR':
        classifier = LogisticRegression()
    elif clf == 'KNN':
        classifier = KNeighborsClassifier()
    elif clf == 'NB':
        classifier = GaussianNB()
    elif clf == 'SVM':
        classifier = SVC()
    classifier.fit(X_train, Y_train)

    # Save the trained model as a pickle string.
    if save_path == None:
        save_path = 'output/classifier_'+clf+'.pkl'
    joblib.dump(classifier, save_path)

    # View the pickled model
    return classifier


def evaluate_classifier(X, Y, classifier_path='output/classifier.pkl'):
    # Load the pickled model
    clf = joblib.load(classifier_path)

    # Use the loaded pickled model to make predictions
    predictions = clf.predict(X)

    print("Prediction :")
    print(np.asarray(predictions, dtype="int32"))
    print("Target :")

    print('Y.shape~ ', Y.shape)
    print(predictions.shape)

    print("\nTest accuration: {}".format(accuracy_score(Y, predictions)))
    print(np.asarray(Y, dtype="int32"))
    print(classification_report(Y, predictions))





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MAHNOB')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    trainParser = subparsers.add_parser('train',
                                        help="Train a new model.")
    trainParser.add_argument('-c', '--clf', type=str, default='NN')
    trainParser.add_argument('-m', '--model', type=str, default=None)
    trainParser.add_argument('-t', '--train_mode', type=int, default=1)


    evalParser = subparsers.add_parser(
        'eval', help='Evaluate a trained model.')
    evalParser.add_argument('-c', '--clf', type=str, default='NN')
    evalParser.add_argument('-m', '--model', type=str)


    args = parser.parse_args()


    if args.mode == 'train':
        data_train = get_training_data(
            '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/'+args.dataset+'/seq_cnn_features/Train', frames, num_classes, input_length)

        if args.clf =='NN':
            model, hist, model_name = train(data=data_train, model_path=args.model, mode=args.train_mode)
            plot_history(hist)
        else:
            x_train, x_val, y_train, y_val = data_train
            print('x_train.shape ', x_train.shape)
            print('x_val.shape ', x_val.shape)
            x_train = np.concatenate((x_train, x_val))
            y_train = np.concatenate((y_train, y_val))
            x_train = x_train.reshape(x_train.shape[0], -1)
            print('x_train.shape ', x_train.shape)
            train_classifier(x_train, y_train, args.clf, save_path='output/classifier_'+args.dataset+'_'+args.clf+'.pkl')


    if args.mode == 'eval':
        # data_test = get_data(
        #     'data/AFEW/seq_cnn_features/Val', frames, num_classes, input_length)
        data_test = get_data(
            '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/'+args.dataset+'/seq_cnn_features/Test', frames, num_classes, input_length)

        if args.clf =='NN':
            # model_path = "checkpoints/MAHNOB_frames_50/bi_lstm-12_loss-0.24555_val_loss-0.25654.h5"
            evaluate(data_test, args.model)
        else:
            X, Y, y, files = data_test
            X = X.reshape(X.shape[0], -1)
            if args.clf == 'SVM':
                Y = y
            evaluate_classifier(X, Y, classifier_path=args.model)


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

    # X_test, Y_test, _, _ = get_data('data/UvA/seq_cnn_features/Train', frames, num_classes, input_length)
    # preds = predict(X_test, model_name)
    # print(preds)
    # print(Y_test)