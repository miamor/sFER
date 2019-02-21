import argparse
import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest
import os


class VGGFaceTests(unittest.TestCase):

    def testVGG16(self):
        keras.backend.set_image_dim_ordering('tf')
        model = VGGFace(model='vgg16')
        img = image.load_img('image/ajb.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)
        preds = model.predict(x)
        print('\n', "VGG16")
        print('\n', x)
        print('\n', preds)
        print('\n', 'Predicted:', utils.decode_predictions(preds))
        self.assertIn('A.J._Buckley', utils.decode_predictions(preds)[0][0][0])
        self.assertAlmostEqual(utils.decode_predictions(preds)[
                               0][0][1], 0.9790116, places=3)

    def testRESNET50(self):
        keras.backend.set_image_dim_ordering('tf')
        model = VGGFace(model='resnet50')
        img = image.load_img('image/ajb.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)
        preds = model.predict(x)
        #print ('\n',"RESNET50")
        # print('\n',preds)
        #print('\n','Predicted:', utils.decode_predictions(preds))
        self.assertIn('A._J._Buckley',
                      utils.decode_predictions(preds)[0][0][0])
        self.assertAlmostEqual(utils.decode_predictions(preds)[
                               0][0][1], 0.91819614, places=3)

    # def testSENET50(self):
    #     keras.backend.set_image_dim_ordering('tf')
    #     model = VGGFace(model='senet50')
    #     img = image.load_img('image/ajb.jpg', target_size=(224, 224))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #     x = utils.preprocess_input(x, version=2)
    #     preds = model.predict(x)
    #     #print ('\n', "SENET50")
    #     #print('\n',preds)
    #     #print('\n','Predicted:', utils.decode_predictions(preds))
    #     self.assertIn(utils.decode_predictions(preds)[0][0][0], 'A._J._Buckley')
    #     self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.91819614)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)

    args = parser.parse_args()

    # unittest.main()
    keras.backend.set_image_dim_ordering('tf')
    model = VGGFace(include_top=False, model='vgg16', input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max

    '''
    img = image.load_img('data/AFEW/Train_AFEW/AlignedFaces_LBPTOP_Points/Faces/000247920/I_1001.jpg', target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)
    preds = model.predict(x)
    print('\n', preds) # vgg features here
    print('\n', preds.shape) # vgg features here
    '''

    print(model.summary())
    
    dataDir = 'data/AFEW/Val_AFEW/AlignedFaces_LBPTOP_Points/Faces/'
    ftDir = 'data/AFEW/Val_AFEW/AlignedFaces_LBPTOP_Points/Features/'
    for folder in os.listdir(dataDir):
        if int(args.start) <= int(folder) <= int(args.end):
            folder_dir = os.path.join(dataDir, folder)
            ft_f_dir = os.path.join(ftDir, folder)
            if not os.path.exists(ft_f_dir):
                os.makedirs(ft_f_dir)

            for img_file in os.listdir(folder_dir):
                ft_file_dir = os.path.join(
                    ft_f_dir, img_file).split('.')[0]+'.txt'

                if not os.path.isfile(ft_file_dir):
                    img_dir = os.path.join(folder_dir, img_file)
                    #print('\nProcessing image '+img_dir)
                    img = image.load_img(img_dir, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = utils.preprocess_input(x, version=1)
                    features = model.predict(x)
                    # save features to file
                    #np.savetxt(img_file.split('.')[0], features, fmt='%d')
                    ft_file_dir = os.path.join(
                        ft_f_dir, img_file).split('.')[0]+'.txt'

                    #print('Saving to '+ft_file_dir)
                    np.savetxt(ft_file_dir, features)

    '''
            for ofile in os.listdir(folder_dir):
                ft_file_dir = os.path.join(ft_f_dir, ofile).split('.')[0]+'.txt'
                ofile_dir = os.path.join(folder_dir, ofile)
                if '.txt' in ofile:
                    print('Moving from '+ofile_dir+' to '+ft_file_dir)
                    os.rename(ofile_dir, ft_file_dir)
    '''