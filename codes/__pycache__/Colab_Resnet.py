'''
Created on Jul 8, 2019

@author: uaa
'''
#https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras 
import numpy as np 
import pandas as pd 
import time
import pickle as cpickle
from keras.callbacks import *
import tensorflow as tf

import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt 

from utils import save_logs

class Classifier_RESNET: 

    def __init__(self, parent_directory, input_shape, nb_classes, verbose=False):
        self.output_directory = parent_directory +"/output"
        self.parent_directory = parent_directory
        self.model = self.build_model(input_shape, nb_classes)
        if(verbose==True):
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory+'model_init.hdf5')
        #------------------------------
        #this block enables GPU enabled multiprocessing
        #https://sefiks.com/2019/03/20/tips-and-tricks-for-gpu-and-multiprocessing-in-tensorflow/
#         core_config = tf.ConfigProto()
#         core_config.gpu_options.allow_growth = True
#         session = tf.Session(config=core_config)
#         keras.backend.set_session(session)

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 8

        input_layer = keras.layers.Input(input_shape)
        
        # BLOCK 1 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum 
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum 
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal 
        shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL 
        
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        #outputlayer bizde sigmoid le olacak, çünkü data binary 
        output_layer = keras.layers.Dense(1, activation='sigmoid')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), 
            metrics=['binary_accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory+'best_model.hdf5' 

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model
    
    def fit(self, x_train, y_train, x_val, y_val,y_true): 
        # x_val and y_val are only used to monitor the test loss and NOT for training  
        batch_size = 4
        nb_epochs = 100
        n_feature_maps = 8 ##onemli build_model altındaki değerle aynı olmalı

        mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

        #start_time = time.time() 

        #hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            #verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

        import pickle
#         pickle_in = open("/home/uaa/eclipse-workspace/GazeModels/output/trainHistoryDict15639552239394848.txt","rb")
#         example_dict = pickle.load(pickle_in)
#         filepath="/content/drive/App/output/epochs:{epoch:03d}-val_acc:{binary_accuracy:.3f}.hdf5"
#         checkpoint = ModelCheckpoint(filepath, monitor='binary_accuracy', verbose=1, save_best_only=True, mode='max')
#         callbacks_list = [checkpoint]
        

        #self.model.load_weights('/content/drive/App/output/Resnet/epochs:003-val_acc:0.886.hdf5') 
        with tf.device('/gpu:0'):
        
#             hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
#             verbose=self.verbose, callbacks=callbacks_list)

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
            verbose=self.verbose, callbacks=self.callbacks)
            # evaluate the model
            scores = self.model.evaluate(x_val, y_val, verbose=0)
        with open(self.parent_directory +'/output/Resnet/score'+str(scores[1] * 100)+'_trnsize'+str(x_train.shape[0])+'_nfmaps'+str(n_feature_maps)+'_bs'+str(batch_size)+'_epochs'+str(nb_epochs), 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)
        #duration = time.time() - start_time

#         model = keras.models.load_model(self.output_directory+'best_model.hdf5')
# 
#         y_pred = model.predict(x_val)
# 
#         # len([i for i in y_pred if i <= 0.5])
#         # convert the predicted from binary to integer 
#         y_pred = np.argmax(y_pred , axis=1)
# 
#         save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()
        
        return scores