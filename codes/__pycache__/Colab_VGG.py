
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2


class CNN1DModel():
    
    global window_size
    window_size=5

    @staticmethod
    def create_model_VGG(nb_input_series=9,nb_outputs=1, 
              optimizer = 'Adam',
              activation = 'relu', nb_filter=32,
              filter_length =3, pool_size1=2, 
              pool_size2=2, dense_param =8 ):

        
        model = Sequential()
        chanDim = -1       
     
            
        model.add(Conv1D(nb_filter=nb_filter, filter_length=filter_length, input_shape=(CNN1DModel.window_size, nb_input_series), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(activation))
        
        model.add(Conv1D(nb_filter=nb_filter, filter_length=filter_length, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(activation))
        
        model.add(MaxPooling1D(pool_size=pool_size1))

        model.add(Conv1D(nb_filter=nb_filter*2, filter_length=filter_length, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(activation))
        
        model.add(Conv1D(nb_filter=nb_filter*2, filter_length=filter_length, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(activation))
        
        model.add(MaxPooling1D(pool_size=pool_size1))
        
        model.add(Conv1D(nb_filter=nb_filter*4, filter_length=filter_length, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(activation))
        
        model.add(Conv1D(nb_filter=nb_filter*4, filter_length=filter_length, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(activation))
        
        model.add(Conv1D(nb_filter=nb_filter*4, filter_length=filter_length, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), padding='same'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation(activation))
        
        model.add(MaxPooling1D(pool_size=pool_size1))
        model.add(Dropout(0.2))
        model.add(Flatten())
        
        model.add(Dense(16, kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Dense(8,  kernel_regularizer=l2(0.01), activation='relu'))
        model.add(Dense(nb_outputs))
        model.add(Activation("sigmoid"))
        
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        
        # return the constructed network architecture
        return model
    