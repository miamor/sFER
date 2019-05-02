from openface.data import iterImgs
import openface.helper
import openface
import random
import sys
import pickle
from tqdm import tqdm
import os
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical

#from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest


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


def get_data(seq_cnn_ft_folder_path, num_frames, num_classes, input_length):
    X = []
    y = []
    files = []

    # read through each video's features file
    for seq_cnn_ft_vid_file in os.listdir(seq_cnn_ft_folder_path):
        z = seq_cnn_ft_vid_file.split('__')
        label = z[0]
        video = z[1]

        ft_file = os.path.join(seq_cnn_ft_folder_path, seq_cnn_ft_vid_file)

        # get cnn features of each video
        temp_list = []
        with open(ft_file, 'rb') as ftin:
            features = pickle.load(ftin)
            i = 0
            for ft in features:
                i = i+1
                if i <= num_frames:
                    temp_list.append(ft)
        flat = np.array(list(temp_list))

        # pad to ft array
        if i < num_frames:
            zeros = np.zeros((num_frames-i, 512))
            flat = np.concatenate((flat, zeros), axis=0)

        X.append(flat)
        y.append(label)
        files.append(video)

    X = np.array(X)
    Y = to_categorical(y, num_classes)

    print(X.shape)
    print(Y.shape)

    return X, Y, np.array(y), files


def get_training_data(seq_folder_path, num_frames, num_classes, input_length):
    ''' get_training_data('data/seq_cnn_features', 40, 2, 512) '''
    X, y, _, _ = get_data(
        seq_folder_path, num_frames, num_classes, input_length)

    # Split into train and test.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42)

    return X_train, X_val, y_train, y_val


def get_test_data(seq_folder_path, num_frames, num_classes, input_length):
    X, y, _, _ = get_data(
        seq_folder_path, num_frames, num_classes, input_length)

    return X, y


def read_spatial_features(ft_vid_folder):
    ''' Read cnn features of all frames (of one video), given the video's features folder '''

    cnn_features = []
    if not os.path.exists(ft_vid_folder):
        print(ft_vid_folder+' not exists. Skip.')
    else:
        for frame_ft_file in os.listdir(ft_vid_folder):
            frame_features = np.loadtxt(ft_vid_folder+'/'+frame_ft_file)
            frame_features = np.array(frame_features)
            # print(frame_features)
            #cnn_features = np.concatenate(cnn_features, frame_features)
            cnn_features.append(frame_features)

    return cnn_features


def create_seq_ft(input_dir, out_dir):
    ''' Merge cnn features of all frames (of one video) into one file containing sequences of cnn features '''

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for label in os.listdir(input_dir):
        y = 1 if label == 'Genuine' else 0

        label_path = os.path.join(input_dir, label)
        # for each video in this label folder
        for folder in os.listdir(label_path):
            folder_path = os.path.join(label_path, folder)

            cnn_features = read_spatial_features(folder_path)

            print(len(cnn_features))
            # print(cnn_features.shape)

            ft_file = os.path.join(out_dir, str(
                y)+'__' + folder + '__'+str(len(cnn_features))+'.pkl')
            if not os.path.isfile(ft_file):
                with open(ft_file, 'wb') as fout:
                    pickle.dump(cnn_features, fout)
            else:
                print('File '+ft_file+' existed.')


def extract_frames(video_path, extract_frames_dir):
    ''' Extract frames from each video and save it to extract_frames_dir '''

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_FPS, 5)
    frames_num = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(str(frames_num)+' frames, fps='+str(fps))

    #success, image = vidcap.read()
    count = 0
    sec = 0
    frameRate = 0.5  # it will capture image in each 0.5 second
    if not os.path.exists(extract_frames_dir):
        os.makedirs(extract_frames_dir)

    success = True
    while success:
        sec = sec + frameRate
        sec = round(sec, 2)
        # added this line
        # vidcap.set(cv2.CAP_PROP_POS_MSEC, (sec*1000))

        success, image = vidcap.read()
        print('Read a new frame: ', success)

        if success:
            file_name = "%00d.png" % count
            file_path = os.path.join(extract_frames_dir, file_name)
            cv2.imwrite(file_path, image)     # save frame as JPEG file
            # time.sleep(1) # seconds

            count += 1

        cv2.waitKey(10)


def align_face(inputDir, outputDir, landmarks='outerEyesAndNose', size=96, useCNN=True, skipMulti=False, verbose=True):
    ''' Align all faces chips from folder inputDir and save output to outputDir '''

    fileDir = '/media/tunguyen/Devs/DeepLearning/FaceReg/openface'
    modelDir = os.path.join(fileDir, 'models')
    dlibModelDir = os.path.join(modelDir, 'dlib')
    # openfaceModelDir = os.path.join(modelDir, 'openface')

    # dlibFacePredictor = os.path.join(
    #     dlibModelDir, "shape_predictor_68_face_landmarks.dat")
    dlibFacePredictor = os.path.join(
        dlibModelDir, "shape_predictor_5_face_landmarks.dat")
    cnnDetectModel = os.path.join(dlibModelDir, "mmod_human_face_detector.dat")

    openface.helper.mkdirP(outputDir)

    imgs = list(iterImgs(inputDir))

    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP,
        '5_landmarks': openface.AlignDlib.BASIC_5_POINTS
    }
    if landmarks not in landmarkMap:
        raise Exception("Landmarks unrecognized: {}".format(landmarks))

    if "predictor_5" in dlibFacePredictor:
        landmarks = '5_landmarks'

    landmarkIndices = landmarkMap[landmarks]

    align = openface.AlignDlib(dlibFacePredictor, cnnDetectModel)

    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        #outDir = os.path.join(outputDir, imgObject.cls)
        # openface.helper.mkdirP(outDir)
        outDir = outputDir
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"
        print('\n'+outDir)

        if os.path.isfile(imgName):
            if verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            #bgr_img = cv2.imread(imgObject.path)
            #rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            if rgb is None:
                if verbose:
                    print("  + Unable to load.")
                outRgb = None
            else:
                if "predictor_68" in dlibFacePredictor:
                    outRgb = align.align(size, rgb,
                                         landmarkIndices=landmarkIndices,
                                         skipMulti=skipMulti,
                                         useCNN=useCNN)
                else:
                    outRgb = align.align(size, rgb,
                                         useCNN=useCNN,
                                         landmarksNum=5)

                if outRgb is None and verbose:
                    print("  + Unable to align.")

            # print(outRgb)
            if outRgb is not None:
                if verbose:
                    print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)


def get_cnn_features(session_path, out_dir, model):
    ''' Extract cnn features using model '''

    for img_file in os.listdir(session_path):
        ft_file_dir = os.path.join(
            out_dir, img_file).split('.')[0]+'.txt'

        if not os.path.isfile(ft_file_dir):
            img_dir = os.path.join(session_path, img_file)
            #print('\nProcessing image '+img_dir)
            img = image.load_img(img_dir, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=1)
            features = model.predict(x)
            # save features to file
            #np.savetxt(img_file.split('.')[0], features, fmt='%d')
            ft_file_dir = os.path.join(
                out_dir, img_file).split('.')[0]+'.txt'

            #print('Saving to '+ft_file_dir)
            np.savetxt(ft_file_dir, features)
