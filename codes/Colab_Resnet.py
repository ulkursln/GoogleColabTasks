'''
Created on Jul 8, 2019

@author: uaa
'''
#https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

# ResNet
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import keras 
from keras.regularizers import l2

class Classifier_RESNET: 

    def __init__(self, parent_directory, input_shape, n_feature_maps, output_drctry):
        self.output_directory = output_drctry
        self.parent_directory = parent_directory
        self.model = self.build_model(input_shape, n_feature_maps)
        self.model.summary()
        self.model.save_weights(self.output_directory+'model_init.hdf5')


    def build_model(self, input_shape, n_feature_maps):
        

        input_layer = keras.layers.Input(input_shape)
        
        # BLOCK 1 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=7, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(input_layer)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
 
        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum 
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(input_layer)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=7, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(output_block_1)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum 
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=1, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(output_block_1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3 

        conv_x = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=7, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(output_block_2)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=5, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps*2, kernel_size=3, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal 
        shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL 
        
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        dropout_layer=keras.layers.Dropout(0.2)(gap_layer, training=True)

        #outputlayer bizde sigmoid le olacak, çünkü data binary 
        output_layer = keras.layers.Dense(1, activation='sigmoid')(dropout_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), 
            metrics=['binary_accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory+'best_model.hdf5' 

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', 
            save_best_only=True)

        self.callbacks = [reduce_lr,model_checkpoint]

        return model
    
    
