import os
from glob import glob
from os.path import join

import numpy as np
from natsort import natsorted
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.layers import Conv2D
from keras.models import Model

import time
import GPUtil


from Datagen import PngClassDataGenerator, PngDataGenerator, NiftiClassDataGenerator, NiftiDataGenerator
from Datagen_albumentations import (PngClassDataGenerator_albumen, PngDataGenerator_albumen, 
                                    NiftiClassDataGenerator_albumen, NiftiDataGenerator_albumen)


# datagen parameters
def get_train_params(batch_size, dims, n_channels, shuffle=True):
    return {'batch_size': batch_size,
            'dim': dims,
            'n_channels': n_channels,
            'shuffle': shuffle,
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'brightness_range': None,
            'shear_range': 0.,
            'zoom_range': 0.15,
            'channel_shift_range': 0.,
            'fill_mode': 'constant',
            'cval': 0.,
            'horizontal_flip': False,
            'vertical_flip': False,
            'rescale': None,
            'preprocessing_function': 'None',
            'interpolation_order': 1}

def get_val_params(batch_size, dims, n_channels, shuffle=False):
    return {'batch_size': batch_size,
            'dim': dims,
            'n_channels': n_channels,
            'shuffle': shuffle,
            'rotation_range': 0,
            'width_shift_range': 0.,
            'height_shift_range': 0.,
            'brightness_range': None,
            'shear_range': 0.,
            'zoom_range': 0.,
            'channel_shift_range': 0.,
            'fill_mode': 'constant',
            'cval': 0.,
            'horizontal_flip': False,
            'vertical_flip': False,
            'rescale': None,
            'preprocessing_function': 'None',
            'interpolation_order': 1}    


    
def get_train_params_albumen(batch_size, dims, n_channels, shuffle=True):
    return {'batch_size': batch_size,
            'dim': dims,
            'n_channels': n_channels,
            'shuffle': shuffle,
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': 0.15,
            'horizontal_flip': False,
            'vertical_flip': False,
            'rescale': None,
            'preprocessing_function': 'None',
            'downscale': 0.0,
            'gauss_noise': 0.0,
            'elastic_transform' : False}

    


def get_val_params_albumen(batch_size, dims, n_channels, shuffle=False):
    return {'batch_size': batch_size,
            'dim': dims,
            'n_channels': n_channels,
            'shuffle': shuffle,
            'rotation_range': 0,
            'width_shift_range': 0.,
            'height_shift_range': 0.,
            'horizontal_flip': False,
            'vertical_flip': False,
            'rescale': None,
            'preprocessing_function': 'None',
             'downscale': 0.0,
            'gauss_noise': 0.0,
            'elastic_transform' : False}


def get_class_datagen(pos_datapath, neg_datapath, train_params, val_params, val_split):
    # Get list of files
    positive_img_files = natsorted(glob(join(pos_datapath, '*.png')))
    print('Found {} positive files'.format(len(positive_img_files)))
    negative_img_files = natsorted(
        glob(join(neg_datapath, '*.png')))
    print('Found {} negative files'.format(len(negative_img_files)))

    # make labels
    pos_labels = [1.]*len(positive_img_files)
    neg_labels = [0.]*len(negative_img_files)
    # combine
    train_img_files = positive_img_files + negative_img_files
    train_labels = pos_labels + neg_labels

    # get class weights for  balancing
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(train_labels), train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Split into test/validation sets
    rng = np.random.RandomState(seed=1)
    trainX, valX, trainY, valY = train_test_split(
        train_img_files, train_labels, test_size=val_split, random_state=rng, shuffle=True)

    train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
    val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

    # Setup datagens
    train_gen = PngClassDataGenerator(trainX,
                                      train_dict,
                                      **train_params)
    val_gen = PngClassDataGenerator(valX,
                                    val_dict,
                                    **val_params)
    return train_gen, val_gen, class_weight_dict


def get_seg_datagen(img_datapath, mask_datapath, train_params, val_params, val_split):

    train_img_files = natsorted(glob(join(img_datapath, '*.png')))
    train_mask_files = natsorted(glob(join(mask_datapath, '*.png')))

    rng = np.random.RandomState(seed=1)
    # Split into test/validation sets
    trainX, valX, trainY, valY = train_test_split(
        train_img_files, train_mask_files, test_size=val_split, random_state=rng, shuffle=True)

    train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
    val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

    # Setup datagens
    train_gen = PngDataGenerator(trainX,
                                 train_dict,
                                 **train_params)
    val_gen = PngDataGenerator(valX,
                               val_dict,
                               **val_params)
    return train_gen, val_gen


def get_class_datagen_albumen(pos_datapath, neg_datapath, train_params, val_params, val_split):
    # Get list of files
    positive_img_files = natsorted(glob(join(pos_datapath, '*.png')))
    print('Found {} positive files'.format(len(positive_img_files)))
    negative_img_files = natsorted(
        glob(join(neg_datapath, '*.png')))
    print('Found {} negative files'.format(len(negative_img_files)))

    # make labels
    pos_labels = [1.]*len(positive_img_files)
    neg_labels = [0.]*len(negative_img_files)
    # combine
    train_img_files = positive_img_files + negative_img_files
    train_labels = pos_labels + neg_labels

    # get class weights for  balancing
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(train_labels), train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Split into test/validation sets
    rng = np.random.RandomState(seed=1)
    trainX, valX, trainY, valY = train_test_split(
        train_img_files, train_labels, test_size=val_split, random_state=rng, shuffle=True)

    train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
    val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

    # Setup datagens
    train_gen = PngClassDataGenerator_albumen(trainX,
                                      train_dict,
                                      **train_params)
    val_gen = PngClassDataGenerator_albumen(valX,
                                    val_dict,
                                    **val_params)
    return train_gen, val_gen, class_weight_dict


def get_seg_datagen_albumen(img_datapath, mask_datapath, train_params, val_params, val_split):

    train_img_files = natsorted(glob(join(img_datapath, '*.png')))
    train_mask_files = natsorted(glob(join(mask_datapath, '*.png')))

    rng = np.random.RandomState(seed=1)
    # Split into test/validation sets
    trainX, valX, trainY, valY = train_test_split(
        train_img_files, train_mask_files, test_size=val_split, random_state=rng, shuffle=True)

    train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
    val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

    # Setup datagens
    train_gen = PngDataGenerator_albumen(trainX,
                                 train_dict,
                                 **train_params)
    val_gen = PngDataGenerator_albumen(valX,
                               val_dict,
                               **val_params)
    return train_gen, val_gen


def get_class_datagen_3d(pos_datapath, neg_datapath, train_params, val_params, val_split):
    # determine if input channels are consistent
    # if train_params['n_channels'] != len(pos_datapath[0]):
    #     print('Ionconsistency between train_params.n_channels and dimensions of data path list -- exiting!')
    #     exit()

    # Get list of files
    positive_img_files = natsorted(pos_datapath)
    print('Found {} positive files'.format(len(positive_img_files)))
    negative_img_files = natsorted(neg_datapath)
    print('Found {} negative files'.format(len(negative_img_files)))

    # make labels
    pos_labels = [1.] * len(positive_img_files)
    neg_labels = [0.] * len(negative_img_files)
    # combine
    train_img_files = positive_img_files + negative_img_files
    train_labels = pos_labels + neg_labels

    # get class weights for  balancing
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(train_labels), train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Split into test/validation sets
    rng = np.random.RandomState(seed=1)
    trainX, valX, trainY, valY = train_test_split(
        train_img_files, train_labels, test_size=val_split, random_state=rng, shuffle=True)

    train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
    val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

    # Setup datagens
    train_gen = NiftiClassDataGenerator(trainX,
                                                train_dict,
                                                **train_params)
    val_gen = NiftiClassDataGenerator(valX,
                                              val_dict,
                                              **val_params)
    return train_gen, val_gen, class_weight_dict




def get_class_datagen_3d_albumen(pos_datapath, neg_datapath, train_params, val_params, val_split):
    
    #determine if input channels are consistent
    if train_params['n_channels'] != len(pos_datapath[0]) :
        print('Ionconsistency between train_params.n_channels and dimensions of data path list -- exiting!')
        exit()
    
    # Get list of files
    positive_img_files = natsorted(pos_datapath)
    print('Found {} positive files'.format(len(positive_img_files)))
    negative_img_files = natsorted(neg_datapath)
    print('Found {} negative files'.format(len(negative_img_files)))

    # make labels
    pos_labels = [1.]*len(positive_img_files)
    neg_labels = [0.]*len(negative_img_files)
    # combine
    train_img_files = positive_img_files + negative_img_files
    train_labels = pos_labels + neg_labels

    # get class weights for  balancing
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(train_labels), train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Split into test/validation sets
    rng = np.random.RandomState(seed=1)
    trainX, valX, trainY, valY = train_test_split(
        train_img_files, train_labels, test_size=val_split, random_state=rng, shuffle=True)

    train_dict = dict([(f, mf) for f, mf in zip(trainX, trainY)])
    val_dict = dict([(f, mf) for f, mf in zip(valX, valY)])

    # Setup datagens
    train_gen = NiftiClassDataGenerator_albumen(trainX,
                                      train_dict,
                                      **train_params)
    val_gen = NiftiClassDataGenerator_albumen(valX,
                                    val_dict,
                                    **val_params)
    return train_gen, val_gen, class_weight_dict



def WaitForGPU(wait=300):
    GPUavailable = False
    while not GPUavailable:
        try:
            if not 'DEVICE_ID' in locals():
                DEVICE_ID = GPUtil.getFirstAvailable()[0]
                print('Using GPU', DEVICE_ID)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
            GPUavailable = True
            return
        except Exception as e:
            # No GPU available
            print('Waiting for GPU...')
            GPUavailable = False
            time.sleep(wait)