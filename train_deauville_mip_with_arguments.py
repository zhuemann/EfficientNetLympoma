# %% Setup

'''
input: 
    model_name  (string)  -- determines which model to use:
                                    'resunet', 'block2d'
    model_num  (string)  --  for saving the model 

    skip_positive_only_training (bool)  -- Skip pretraining segmentation model 
                                     on only images that are positive?                                     
    skip_all_cases_training (bool) -- don't train on all cases (the default 
                                    order is all-train, positive-train, 
                                    quality-train)
    skip_quality_only_training (bool)  -- don't train on high quality positive 
                                    cases
    skip_encoder_pretrain (bool) -- skip the pre-training of the classification 
                                   model on NIH data
    skip_encoder_training -- skip any training of classification model/encoder
    




'''

import os
import time
from glob import glob
from os.path import join
import sys

from matplotlib import pyplot as plt
import GPUtil
import numpy as np
import tensorflow as tf
from keras import models

os.environ['TF_KERAS'] = '1'
import os


from tensorflow.python.keras.backend import set_session
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

#from keras.optimizers import Adam


#from natsort import natsorted
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras_radam import RAdam
from keras.optimizers import adam_v2


#from CustomCallbacks import CyclicLR

#from Datagen_albumentations import PngClassDataGenerator_albumen, PngDataGenerator_albumen
from HelperFunctions import (get_class_datagen, get_class_datagen_albumen,
                             get_train_params, get_val_params,
                             get_train_params_albumen, get_val_params_albumen)
#from Losses import dice_coef_loss
from Models import (BlockModel2D, BlockModel_Classifier, ConvertEncoderToCED, 
                                res_unet, res_unet_encoder, attention_unet, 
                                tiramisu, efficientB7_model, efficientB4_model,
                                 efficientB0_model)

import argparse

###########################  Functions  ######################################


def get_seg_model(model_name, input_dims):
    #return segmentation model based on name
    if model_name.lower() == 'block2d':
        full_model = BlockModel2D(input_dims, filt_num=16, numBlocks=4)
    elif model_name.lower() == 'resunet':
        full_model = res_unet(input_dims)
    elif model_name.lower() == 'attunet':
        full_model = attention_unet(input_dims)
    elif model_name.lower() == 'tiramisu':
        full_model = tiramisu(input_dims)
    
    return full_model

def get_classifier_model(model_name, input_dims):
    #return segmentation model based on name
    if model_name.lower() == 'block2d':
        full_model = BlockModel_Classifier(input_shape=input_dims,
                                          filt_num=16, numBlocks=4)
    elif model_name.lower() == 'resunet':
        full_model = res_unet_encoder(input_dims)
        
    elif model_name.lower() == 'efficientb7':
        full_model = efficientB7_model(input_dims)
        
    elif model_name.lower() == 'efficientb4':
        full_model = efficientB4_model(input_dims)
        
    elif model_name.lower() == 'efficientb0':
        full_model = efficientB0_model(input_dims)        
    
    else:
        full_model = 'error! no model!'
        print(full_model)
        
    
    return full_model



##############################################################################
def str2bool(v):   # for parsing booleans -- which are a pain!
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_deauville_mip_with_arguments(model_name = 'efficientb7', model_num = 1):
    # DEVICE_ID_LIST = GPUtil.getFirstAvailable()
    # DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
    os.chdir('Z:/Lymphoma_UW_Retrospective/Codes/')
    ## parse inputs

    #parser = argparse.ArgumentParser(description='Model and parameters')
    #parser.add_argument('model_name', type=str, help='Model type/name')
    #parser.add_argument('model_num', type=str, help='Model save number')
    #parser.add_argument('start_new', type=str2bool, nargs='?', const=True, default=True, help='Start from scratch (otherwise load model)?')

    #args = parser.parse_args(args)


    #model_name = args.model_name
    #model_num = args.model_num
    #start_new = args.start_new
    start_new = True
    #!!!
    #model_name = 'efficientB0'
    #model_num = '2'
    #start_new = 1

    config = tf.compat.v1.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = False
    sess = tf.compat.v1.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)

    os.environ['HDF5_USE_FILE_LOCKING'] = 'false'


    rng = np.random.RandomState(seed=1)



    # ~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~ SETUP~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~

    # Setup data
    pos_train_datapath = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'
    neg_train_datapath = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'

    best_model_filepath = 'deauville_mip_{}_v{}.h5'

    # train parameters
    im_dims = (768,768)
    n_channels = 1
    batch_size = 4

    learnRate = 1e-4
    val_split = .15
    test_split = .15
    epochs = [30, 40]  #each addition loads the previous best model, increases training rate and tries again



    # datagen params
    train_params = get_train_params_albumen(batch_size, im_dims, n_channels)
    val_params = get_val_params_albumen(batch_size, im_dims, n_channels)


    #I've saved the preprocessed data with clahe, so remove it
    #pre_train_params["preprocessing_function"] = 'None'
    #pre_val_params["preprocessing_function"] = 'None'
    train_params['width_shift_range'] = 0.1
    train_params['height_shift_range'] = 0.1
    train_params['rescale'] = 65535.    #note DEFAULT = 255
    train_params['zoom_range'] = 0.1
    train_params['downscale'] = 0.25
    train_params['gauss_noise'] = 0.005
    train_params['gauss_blur'] = 11
    train_params['elastic_transform'] = False #was true
    train_params['elastic_transform_params'] = (100,10,10)

    val_params['rescale'] = 65535.




    # %% ~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~Classification_trainer~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~

    # Get datagens for pre-training
    train_gen, val_gen, class_weights = get_class_datagen_albumen(
        pos_train_datapath, neg_train_datapath, train_params, val_params, val_split)

    [test_im, test_lab] = train_gen.__getitem__(1)
    print('---\n---Max value of test image is ' + str(np.max(test_im)) )
    print('---Label of test image is ' + str(test_lab[0]) + '\n---')
    plt.imshow(test_im[0,:,:,0])
    plt.pause(3)
    plt.close()


    if start_new:
        # Create model
        model = get_classifier_model(model_name, im_dims+(n_channels,))

        # Compile model
        # model.compile(RAdam(), loss='binary_crossentropy', metrics=['accuracy'])
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        print('\nLoading model ' + best_model_filepath.format(model_name, model_num) + '\n')
        model = models.load_model(best_model_filepath.format(model_name, model_num), custom_objects={'RAdam': RAdam})

    # Create callbacks
    cb_check = ModelCheckpoint(best_model_filepath.format(model_name, model_num), monitor='val_loss',
                               verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #cb_plateau = ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=5, verbose=1)

    print('---------------------------------')
    print('----- Starting training -----')
    print('---------------------------------')

    for i, epochs_i in enumerate(epochs):
        print('Training round ' + str(i))

        # Train model
        history = model.fit_generator(generator=train_gen,
                                              epochs=epochs_i, verbose=1,
                                              callbacks=[cb_check], #callbacks=[cb_check, cb_plateau],
                                              class_weight=class_weights,
                                              validation_data=val_gen)
        # Plot training & validation accuracy values
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        #load best one before starting again
        model = models.load_model(best_model_filepath.format(model_name, model_num), custom_objects={'RAdam': RAdam})


    # Load best weights
    #model = models.load_model(best_model_filepath.format(model_name, model_num))

    # Calculate confusion matrix
    print('Calculating classification confusion matrix...')
    val_gen.shuffle = False
    preds = model.predict_generator(val_gen, verbose=1)
    labels = [val_gen.labels[f] for f in val_gen.list_IDs]
    y_pred = np.rint(preds)
    totalNum = len(y_pred)
    y_true = np.rint(labels)[:totalNum]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('----------------------')
    print('Classification Results')
    print('----------------------')
    print('True positives: {}'.format(tp))
    print('True negatives: {}'.format(tn))
    print('False positives: {}'.format(fp))
    print('False negatives: {}'.format(fn))
    print('% Positive: {:.02f}'.format(100*(tp+fp)/totalNum))
    print('% Negative: {:.02f}'.format(100*(tn+fn)/totalNum))
    print('% Accuracy: {:.02f}'.format(100*(tp+tn)/totalNum))
    print('-----------------------')

    thresh = 0.3
    y2 = np.copy(preds)
    s = preds <= thresh
    s2 = preds > thresh
    y2[s] = 0
    y2[s2] = 1
    y_pred = np.rint(y2)


#    else:
#        # Just create model, then load weights
#        model = get_classifier_model(model_name, im_dims+(n_channels,))
#        # Load best weights
#        model.load_weights(pretrain_weights_filepath.format(model_name, model_num))
#    
#    # %% ~~~~~~~~~~~~~~~~~~~~~~~
#    # ~~~~~~ Training ~~~~~~~
#    # ~~~~~~~~~~~~~~~~~~~~~~~
#    
#    print('Setting up 512-training')
#    
#    # convert to segmentation model
#    model = ConvertEncoderToCED(model, model_name)
#    
#    # create segmentation datagens
#    # using positive, large mask images only
#    train_gen, val_gen = get_seg_datagen(
#        pos_img_filt_path, pos_mask_filt_path, train_params, val_params, val_split)
#    
#    
#    # Create callbacks
#    best_weight_path = best_weight_filepath.format(model_name,'512train', model_num)
#    cb_check = ModelCheckpoint(best_weight_path, monitor='val_loss',
#                               verbose=1, save_best_only=True,
#                               save_weights_only=True, mode='auto', period=1)
#    
#    cb_plateau = ReduceLROnPlateau(
#        monitor='val_loss', factor=.5, patience=3, verbose=1)
#    
#    # Compile model
#    model.compile(Adam(lr=learnRate), loss=dice_coef_loss)
#    
#    print('---------------------------------')
#    print('----- Starting 512-training -----')
#    print('---------------------------------')
#    
#    history = model.fit_generator(generator=train_gen,
#                                  epochs=epochs_unfreeze[0], verbose=1,
#                                  callbacks=[cb_plateau],
#                                  validation_data=val_gen)
#    
#    # make all layers trainable again
#    for layer in model.layers:
#        layer.trainable = True
#    
#    # Compile model
#    model.compile(Adam(lr=learnRate), loss=dice_coef_loss)
#    
#    print('----------------------------------')
#    print('--Training with unfrozen weights--')
#    print('----------------------------------')
#    
#    history2 = model.fit_generator(generator=train_gen,
#                                   epochs=epochs_unfreeze[1],
#                                   verbose=1,
#                                   callbacks=[cb_check, cb_plateau],
#                                   validation_data=val_gen)
#    
#    # %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    # ~~~~~~ Full Size Training ~~~~~~~
#    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    
#    
#    print('Setting up 1024 training')
#    
##    # make full-size model
##    if model_name.lower() == 'blockmodel2d':
#    input_dims = (1024, 1024, n_channels)
#    full_model = get_seg_model(model_name, input_dims)
#    full_model.load_weights(best_weight_path)
#
#
#else:
#    input_dims = (1024, 1024, n_channels)
#    full_model = get_seg_model(model_name, input_dims)    
#
#    
#    
## Compile model
#full_model.compile(Adam(lr=learnRate), loss=dice_coef_loss)
#
## Set weight paths
#best_weight_path = best_weight_filepath.format(model_name,'1024train', model_num)
#
#
## Create callbacks
#cb_plateau = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=5, verbose=1)
#cb_check = ModelCheckpoint(best_weight_path, monitor='val_loss',
#                           verbose=1, save_best_only=True,
#                           save_weights_only=True, mode='auto', period=1)    
##train
#
##train with all data
#if  all_cases_training:
#    print('Training with all data')
#    history_full, full_model, train_gen, val_gen  = get_data_and_train_model(full_model, 
#                                    all_img_path, all_mask_path, 
#                                    full_train_params, full_val_params, val_split, 
#                                    best_weight_path, cb_check, learnRate, 
#                                    full_epochs, cb_plateau)
#
##if post-training with only positive masks
#if positive_only_training:
#    print('Training with only positives...')
#    history_full, full_model, train_gen, val_gen  = get_data_and_train_model(full_model, 
#                                pos_img_path, pos_mask_path, 
#                                full_train_params, full_val_params, val_split, 
#                                best_weight_path, cb_check, learnRate, 
#                                full_epochs, cb_plateau)
#
#
#if  quality_only_training:
#    print('Training with only quality positives...')
#    history_full, full_model, train_gen, val_gen  = get_data_and_train_model(full_model, 
#                                qual_img_path, qual_mask_path, 
#                                full_train_params, full_val_params, val_split, 
#                                best_weight_path, cb_check, learnRate, 
#                                full_epochs, cb_plateau)
#
#
## %% make some demo images
#
#full_model.load_weights(best_weight_path)
#
#folder_save='sample_difference_images_{}_v{}'.format(model_name, model_num)
#if not os.path.exists(folder_save):
#    os.mkdir(folder_save)
#
#count = 0
#for rep in range(20):
#    testX, testY = val_gen.__getitem__(rep)
#    preds = full_model.predict_on_batch(testX)
#
#    for im, mask, pred in zip(testX, testY, preds):
#        DisplayDifferenceMask(im[..., 0], mask[..., 0], pred[..., 0],
#                              savepath=os.path.join(folder_save, 
#                                'SampleDifferenceMasks_{}.png'.format(count)))
#        count += 1
#
#
## %%
