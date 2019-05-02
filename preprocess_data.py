import sys
import pickle
import os
import numpy as np
import glob
import keras

from util import *
from keras_vggface import VGGFace

import xml.etree.ElementTree as ET

from shutil import copyfile
import pandas


def filter_mahnob():
    '''
    Filter laughing/smiling frames only (remove speaking frames)
    '''
    # read laughterAnnotation file

    dDir = '/media/tunguyen/Others/Dataset/FacialExpressions/MAHNOB/Labeled/Sessions/'
    out_dir = '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/MAHNOB/frames'
    new_out_dir = '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/MAHNOB/frames_filter'

    for expression__session in sorted(os.listdir(dDir)):
        z = expression__session.split('__')
        expression = z[0]
        session = z[1]
        extract_frames_dir = os.path.join(out_dir, expression+'/'+session)
        if 1 == 1:
            folder_path = os.path.join(dDir, expression__session)
            print(folder_path+' ================')
            ano_filename = 'laughterAnnotation.csv'
            ano_file_path = os.path.join(folder_path, ano_filename)

            df = pandas.read_csv(ano_file_path,
                                 header=0,
                                 names=['Excluded', 'Type', 'Start Time (sec)', 'End Time (sec)', 'Start Frame', 'End Frame'])
            print(df)
            rows = df.shape[0]

            for i in range(rows):
                type = df.iloc[i]['Type']
                start = df.iloc[i]['Start Frame']
                end = df.iloc[i]['End Frame']
                
                print(str(i)+' - '+type)

                if type in ['PosedLaughter', 'Laughter', 'SpeechLaughter', 'Speech']:
                # if type in ['SpeechLaughter']:
                    # cls = '0' if type == 'PosedLaughter' else '1'
                    if type == 'PosedLaughter':
                        cls = '0'
                    elif type == 'Laughter':
                        cls = '1'
                    elif type == 'SpeechLaughter':
                        cls = '1_'
                    elif type == 'Speech':
                        cls = '0_'

                    new_frames_dir = os.path.join(new_out_dir, 
                        cls+'/'+session+'_'+str(start)+'_'+str(end))
                    if not os.path.exists(new_frames_dir):
                        os.makedirs(new_frames_dir)

                    frame = start
                    while start <= frame <= end:
                        frame_path = os.path.join(
                            extract_frames_dir, str(frame)+'.png')
                        new_frame_path = os.path.join(
                            new_frames_dir, str(frame)+'.png')
                        if not os.path.exists(new_frame_path):
                            copyfile(frame_path, new_frame_path)
                        frame += 1


def extract_frames_dataset(dataset):
    if dataset == 'AFEW':
        extract_frames_afew()
    elif dataset == 'MMI':
        extract_frames_mmi()
    elif dataset == 'MAHNOB':
        extract_frames_mahnob()


def extract_frames_afew():
    # dDir = '/media/tunguyen/Others/Dataset/FacialExpressions/'+dataset+'/Val_AFEW/'
    dDir = '/media/tunguyen/Others/Dataset/FacialExpressions/AFEW/videos/'
    out_dir = 'data/AFEW/frames'
    for expression in sorted(os.listdir(dDir)):
        # if expression in ['Sad_']:
        if 1 == 1:
            folder_path = os.path.join(dDir, expression)
            print(folder_path+' ================')
            ext = 'avi'
            # ext = 'mp4'
            for video_path in sorted(glob.glob(os.path.join(folder_path, "*."+ext))):
                extract_frames_dir = os.path.join(
                    out_dir, video_path.split('.'+ext)[0].split(dDir)[1])
                # print('\nSave path: '+extract_frames_dir)
                # extract_frames(video_path, extract_frames_dir)
                if not os.path.exists(extract_frames_dir):
                    print('\nSave path: '+extract_frames_dir)
                    extract_frames(video_path, extract_frames_dir)
                else:
                    print('\n'+extract_frames_dir+' exists')


def extract_frames_mmi():
    dDir = '/media/tunguyen/Others/Dataset/FacialExpressions/MMI/mmi_happy/Sessions/'
    out_dir = '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/MMI/frames'
    for expression__session in sorted(os.listdir(dDir)):
        z = expression__session.split('__')
        expression = z[0]
        session = z[1]
        if 1 == 1:
            folder_path = os.path.join(dDir, expression__session)
            print(folder_path+' ================')
            ext = 'avi'
            # ext = 'mp4'
            for video_path in sorted(glob.glob(os.path.join(folder_path, "*."+ext))):
                extract_frames_dir = os.path.join(
                    out_dir, expression+'/'+session)
                if not os.path.exists(extract_frames_dir):
                    print('\nSave path: '+extract_frames_dir)
                    extract_frames(video_path, extract_frames_dir)
                else:
                    print('\n'+extract_frames_dir+' exists')


def extract_frames_mahnob():
    dDir = '/media/tunguyen/Others/Dataset/FacialExpressions/MAHNOB/Labeled/Sessions/'
    out_dir = '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/MAHNOB/frames'
    for expression__session in sorted(os.listdir(dDir)):
        z = expression__session.split('__')
        expression = z[0]
        session = z[1]
        if 1 == 1:
            folder_path = os.path.join(dDir, expression__session)
            print(folder_path+' ================')
            ext = 'avi'
            # ext = 'mp4'
            for video_path in sorted(glob.glob(os.path.join(folder_path, "*."+ext))):
                extract_frames_dir = os.path.join(
                    out_dir, expression+'/'+session)
                if not os.path.exists(extract_frames_dir):
                    print('\nSave path: '+extract_frames_dir)
                    extract_frames(video_path, extract_frames_dir)
                else:
                    print('\n'+extract_frames_dir+' exists')


def get_label_mmi():
    sessionsDir = '/media/tunguyen/Others/Dataset/FacialExpressions/mmi-facial-expression-database_download_2019-04-27_05_06_23/Sessions'
    outDir = 'data/MMI'
    for session in os.listdir(sessionsDir):
        session_path = os.path.join(sessionsDir, session)

        emotion = ''
        for file in os.listdir(session_path):
            ext = file.split('.')[1]
            '''
            if ext == 'xml' and file != 'session.xml' and (not '-oao_aucs' in file):
                session_xml = os.path.join(session_path, file)
                print(session_xml)

                root = ET.parse(session_xml).getroot()
                # print(root)

                Metatag = root.findall('Metatag')[1]
                for Metatag in root.findall('Metatag'):
                    # print(Metatag)
                    name_tag = Metatag.get('Name')
                    if (name_tag == 'Emotion'):
                        emotion = Metatag.get('Value')
                        print(emotion)

                session_path__new = os.path.join(sessionsDir, emotion+'__'+session)
            '''

            if ext in ['jpg', 'png']:
                # session_path__new = os.path.join(sessionsDir, emotion+'__'+session+'__img')
                session_path__new = os.path.join(sessionsDir, session+'__img')
                os.rename(session_path, session_path__new)

        # os.rename(session_path, session_path__new)


def detect_align_face(dataset):
    '''
    detect and align faces from frames
    loop through the folder of videos,
    sessionsDir contains multiple folder,
    each contains multiple frames extracted from that session
    '''
    frame_dir = '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/'+dataset+'/frames'
    if dataset == 'MAHNOB':
        frame_dir = '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/'+dataset+'/frames_filter'
    alignDir = '/media/tunguyen/Others/Dataset/FacialExpressions/processed_data/' + \
        dataset+'/aligned_faces_extracted'
    for expression in sorted(os.listdir(frame_dir)):
        # if expression in ['Happy', 'Neutral', 'Sad_']:
        if 1 == 1:
            expression_path = os.path.join(frame_dir, expression)
            print(expression_path)
            for inputDir_name in sorted(os.listdir(expression_path)):
                outputDir = os.path.join(
                    alignDir, expression+'/'+inputDir_name)
                if not os.path.exists(outputDir):
                    os.makedirs(outputDir)

                inputDir = os.path.join(expression_path, inputDir_name)
                align_face(inputDir, outputDir)


if __name__ == '__main__':

    ''' MMI dataset '''
    # get_label_mmi()

    # extract frames
    # extract_frames_dataset('AFEW')
    # extract_frames_dataset('MMI')
    # extract_frames_dataset('MAHNOB')
    # filter_mahnob()

    # detect_align_face('MMI')
    detect_align_face('MAHNOB')

    '''
    # detect and align faces from frames
    # loop through the folder of videos,
    # sessionsDir contains multiple folder,
    # each contains multiple frames extracted from that session
    frame_dir = 'data/'+dataset+'/frames'
    alignDir = 'data/'+dataset+'/aligned_faces_extracted'
    for expression in sorted(os.listdir(frame_dir)):
        # if expression in ['Happy', 'Neutral', 'Sad_']:
        if 1==1:
            expression_path = os.path.join(frame_dir, expression)
            for inputDir_name in os.listdir(expression_path):
                outputDir = os.path.join(alignDir, expression+'/'+inputDir_name)
                if not os.path.exists(outputDir):
                    os.makedirs(outputDir)

                inputDir = os.path.join(expression_path, inputDir_name)
                align_face(inputDir, outputDir)


    # alignDir = 'data/'+dataset+'/aligned_faces_extracted'

    # extract cnn features of each aligned face using VGGface
    out_ft = 'data/'+dataset+'/cnn_features_by_frame/'
    keras.backend.set_image_dim_ordering('tf')
    model = VGGFace(include_top=False, model='vgg16', input_shape=(
        224, 224, 3), pooling='avg')  # pooling: None, avg or max
    model.summary()

    for expression in os.listdir(alignDir):
        expression_path = os.path.join(alignDir, expression)

        if expression in ['Happy', 'Neutral', 'Sad_']:
            if expression == 'Happy':
                out_ftDir = os.path.join(out_ft, 'Genuine')
            else:
                out_ftDir = os.path.join(out_ft, 'Fake')
        else:
            out_ftDir = os.path.join(out_ft, expression)

        for session in os.listdir(expression_path):
            session_path = os.path.join(expression_path, session)
            # output folder to save features
            ft_f_dir = os.path.join(out_ftDir, session)
            if not os.path.exists(ft_f_dir):
                os.makedirs(ft_f_dir)

            get_cnn_features(session_path, ft_f_dir, model)


    # create sequence of cnn features and save to
    frame_ft_dir = 'data/'+dataset+'/cnn_features_by_frame/'
    out_seq_ft_dir = 'data/'+dataset+'/seq_cnn_features/Train/'
    create_seq_ft(frame_ft_dir, out_seq_ft_dir)

    '''
