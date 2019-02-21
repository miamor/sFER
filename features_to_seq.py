import tensorflow as tf, sys
import pickle
import sys
from tqdm import tqdm
import os
import numpy as np

fold = 'Val'
features_DIR = 'data/AFEW/'+fold+'_AFEW/AlignedFaces_LBPTOP_Points/Features/'

def create_label_file():
    # Create a file containing list of genuine smiley faces
    labeled_video = []
    for filename in os.listdir('data/AFEW/'+fold+'_AFEW/Happy'):
        filename = filename.split('.')[0]
        labeled_video.append([filename, 1])

    for filename in os.listdir('data/AFEW/'+fold+'_AFEW/Neutral'):
        filename = filename.split('.')[0]
        labeled_video.append([filename, 0])

    with open('data/labeled-video-'+fold+'.pkl', 'wb') as fout:
        pickle.dump(labeled_video, fout)


def get_spatial_features_video(video, label):
    cnn_features = []
    if not os.path.exists(features_DIR+video):
        print(features_DIR+video+' ('+str(label)+') not exists. Skip.')
    else:
        for frame_file in os.listdir(features_DIR+video):
            frame_features = np.loadtxt(features_DIR+video+'/'+frame_file)
            cnn_features.append(frame_features)

    return cnn_features

def save_features():
    with open('data/labeled-video-'+fold+'.pkl', 'rb') as fin:
        labeled_videos = pickle.load(fin) # simply [[video, label],...]
        for video_label in labeled_videos:
            video = video_label[0]
            label = video_label[1]

            #print(video+' ~ '+str(label))
            
            cnn_features = get_spatial_features_video(video, label)

            ft_file = 'data/cnn_features/cnn-features-video-' + video + '.pkl'
            if not os.path.isfile(ft_file):
                with open(ft_file, 'wb') as fout:
                    pickle.dump(cnn_features, fout)
            else:
                print('File '+ft_file+' existed.')
            

if __name__ == '__main__':
    #create_label_file()
    save_features()