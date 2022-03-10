
# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Example of using Keras to implement a 1D convolutional neural network (CNN) for timeseries prediction.
"""

#from __future__ import print_function, division
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
#from model_selection import GridSearch

import sys
sys.path.append('/home/uaa/eclipse-workspace/KerasModels')
import os
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Colab_DataUtils import dataUtils as dataUtils
#from CNN1D import CNN1D as cnn1D, CNN1D
from SimpleCNN1D import SimpleCNN1D as simplecnn1D, SimpleCNN1D

from SimpleMLP import SimpleMLP as simpleMLP
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib.patches import Patch
import numpy
import pickle as cpickle
import tensorflow as tf
import multiprocessing
from multiprocessing import Pool


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense


class hyperTuningParameters(object):
    def __init__(self, window_size, optimizer,
                 activation, nb_filter,filter_length,
                 pool_size1, pool_size2, dense_param,
                 epochs,batch_size, scoreParams, stddevParams):
        
        self.window_size=window_size
        self.optimizer = optimizer
        self.activation = activation
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.pool_size1 = pool_size1
        self.pool_size2 = pool_size2
        self.dense_param = dense_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.scoreParams = scoreParams
        self.stddevParams =stddevParams

scoreList=[]
best_score=0
best_score_parameters= hyperTuningParameters(window_size=0, optimizer=0, activation=0,
                                                                  nb_filter=0, filter_length=0,
                                                                  pool_size1 = 0,pool_size2= 0,
                                                                  dense_param=0,epochs=0, batch_size=0,
                                                                  scoreParams=0, stddevParams=0)


        

  
# #fikir vermesi acısından ,HyperTuning: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
# #https://forums.fast.ai/t/batchnormalization-axis-for-convolution-layers-in-keras/5458

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
#https://github.com/TannerGilbert/Tutorials/blob/master/Keras-Tutorials/8.%20Hyperparameter%20Tuning%20using%20the%20Scikit%20Learn%20Wrapper/Keras%20%238%20-%20Keras%20Hyperparameter%20Tuning%20using%20Scikit%20Learn%20Wrapper.ipynb

#https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py    

def make_timeseries_instances(timeseries, timeseries_y, window_size):
    """Make input features and prediction targets from a `timeseries` for use in machine learning.

    :return: A tuple of `(X, y, q)`.  `X` are the inputs to a predictor, a 3D ndarray with shape
      ``(timeseries.shape[0] - window_size, window_size, timeseries.shape[1] or 1)``.  For each row of `X`, the
      corresponding row of `y` is the next value in the timeseries.  The `q` or query is the last instance, what you would use
      to predict a hypothetical next (unprovided) value in the `timeseries`.
    :param ndarray timeseries: Either a simple vector, or a matrix of shape ``(timestep, series_num)``, i.e., time is axis 0 (the
      row) and the series is axis 1 (the column).
    :param int window_size: The number of samples to use as input prediction features (also called the lag or lookback).
    """
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    y = timeseries_y[window_size:]
    q = np.atleast_3d([timeseries[-window_size:]])
    return X, y, q


#https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py    

def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    #cmap_data = cm.Paired
    #cmap_cv = cm.coolwarm
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=plt.cm.get_cmap('coolwarm'),
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.get_cmap('Paired'))


    # Formatting
    yticklabels = list(range(n_splits)) + ['class']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, 100])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def runMLP(timeseries,timeseries_y):
        mlp=simpleMLP.create_model()
        mlp.summary()
        #KerasClassifier 'a mlp'yi parametre olarak verince hata atıyor o nedenle simpleMLP.create_model , yani () olarak yeniden çağırdım 
        model = KerasClassifier(build_fn=simpleMLP.create_model, verbose=1)
        
        # grid search epochs, batch size and optimizer
        optimizers = ['rmsprop', 'adam']
        init = ['glorot_uniform', 'normal', 'uniform']
        epochs = [50, 100, 150]
        batches = [5, 10, 20]
        param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
        grid = GridSearchCV(estimator=model, param_grid=param_grid)
        grid_result = grid.fit(timeseries, timeseries_y)
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

def simpleCNN1d():

    "Prepare input data, build model, evaluate."
    np.set_printoptions(threshold=25)
   
    
    #LOAD THE DATASET 
    current_directory  = os.path.dirname(os.path.realpath('__file__'))
    parent_directory = os.path.split(current_directory)[0] # Repeat as needed 
    parent_directory="/content/drive/App/"
    
    timeseries, timeseries_y = dataUtils.load_dataset(os.path.join(parent_directory, 'input/ConsolidatedDataLearning.pkl'))
    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    
 
    #runMLP(timeseries,timeseries_y) 
#     
#     
#     batch_sizes = [5,10, 20, 50]
#     epochs = [5, 10, 50]
#     filter_numbers = [2,4,8]
#     pool_size1 = [5,4,3,2]
#     pool_size2 = [4,3,2]
#     filter_length = [2,3,5,7,10]#kernel_size (min fixation count 100 ms olduğu için en az 3 tane almak aslında daha mantıklı geldi, oyle yorumlayabilirisn
#     dense_param = [8,16,32,64]
#       parameters = dict(
#                   optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
#                   activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
#                   nb_filter = filter_numbers, 
#                   filter_length = filter_length,
#                   pool_size1 =pool_size1,
#                   pool_size2 =pool_size2,
#                   dense_param =dense_param,
#                   epochs =epochs,
#                   batch_size = batch_sizes,
#                   
#                   )

    batch_sizes = [4,100]
    epochs = [10,100]
    filter_numbers = [4,16]
    pool_size1 = [2]
    pool_size2 = [2]
    filter_length = [3,5]#kernel_size (min fixation count 100 ms olduğu için en az 3 tane almak aslında daha mantıklı geldi, oyle yorumlayabilirisn
    dense_param = [16]
    parameters = dict(
                  optimizer = ['Adam' ],
                  activation = [ 'relu'],
                  nb_filter = filter_numbers, 
                  filter_length = filter_length,
                  pool_size1 =pool_size1,
                  pool_size2 =pool_size2,
                  dense_param =dense_param,
                  epochs =epochs,
                  batch_size = batch_sizes,
                  
                  )
#     
#     parameters = dict(
#                       optimizer = ['Adam'],
#                       activation = ['relu'],
#                       nb_filter = [2,4], 
#                       filter_length = [1],
#                       pool_size1 =[3],
#                       pool_size2 = [2],
#                       dense_param =[4],
#                       epochs =[5],
#                       batch_size = [10]
#                       
#                       )
   
    #gsearch = GridSearchCV(estimator=model, cv=my_cv,param_grid=parameters)
    

    
    
    #windows_size_arr=[10,15,20, 30]
    #windows_size_arr=[8,10]
    windows_size_arr=[13]
    for ws in windows_size_arr:
        X, y, q = make_timeseries_instances(timeseries, timeseries_y, ws)
        print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q)
        print('\n\nInput shapes:', X.shape, '\n\nOutput shape:', y.shape, '\n\nQuery vector shape:', q.shape)
         
        
        # partition the data into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing
        test_size = int(0.1 * nb_samples)           # In real life you'd want to use 0.2 - 0.5
        X_train_and_validation, X_test, y_train_and_validation, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:] 
 
        for opt in parameters['optimizer']:
            for actv in parameters['activation']:
                for nbfltr in parameters['nb_filter']:
                    for fltrlength in parameters['filter_length']:
                        for pollsize1 in parameters['pool_size1']:
                            for pollsize2 in parameters['pool_size2']:
                                for denseprm in parameters['dense_param']:
                                    for epoch in parameters['epochs']:
                                        for batchsz in parameters['batch_size']:
                                            #build model
                                            #CNN1D.window_size =ws
                                            SimpleCNN1D.window_size =ws
                                            try:
                                                
                                                cnn1dNet = simplecnn1D.create_model(nb_input_series=nb_series, nb_outputs=1,
                                                                          optimizer = opt,
                                                                          activation = actv, nb_filter=nbfltr,
                                                                          filter_length =fltrlength, pool_size1=pollsize1, 
                                                                          pool_size2=pollsize2, dense_param =denseprm)
            
                                                #print('\n\nModel with input size {}, output size {}'.format(cnn1dNet.input_shape, cnn1dNet.output_shape))
                                                cnn1dNet.summary()

                                                
                                                ts_splits = TimeSeriesSplit(n_splits=5)
                                                #plt.figure(1)
                                                #index = 1
                                                #tscv_list=list(ts_splits.split(X_train_and_validation))
                                                cvscores = []
                                                for train_index, validation_index in ts_splits.split(X_train_and_validation):
                                                    training_series= X_train_and_validation[train_index]
                                                    training_series_y= y_train_and_validation[train_index]
                                                    
                                                    validation_series= X_train_and_validation[validation_index]
                                                    validation_series_y= y_train_and_validation[validation_index]
            
                                        #                 fig, ax = plt.subplots()
                                        #                 plot_cv_indices(ts_splits, X_train_and_validation, y_train_and_validation,  ax, n_splits)
                                                    
                                                    print('Observations: %d' % (len(training_series) + len(validation_series)))
                                                    print('Training Observations x,y: %d, %d' %(len(training_series), len(training_series_y)) )
                                                    print('Validation Observations x,y: %d, %d' % (len(validation_series),len(validation_series_y)))
    #                                                 
    #                                                 print('\n\Training shapes:', training_series.shape, '\n\nOutput shape:', training_series_y.shape)
    #                                                 print('\n\Validation shapes:', validation_series.shape, '\n\nOutput shape:', validation_series_y.shape)
    #                                                 
    #                                                 print('\n\nWindow size:', ws, '\n optimizer:', opt, '\n activation: ', actv,
    #                                                       '\n nb_filter:', nbfltr, '\n filter_length:', fltrlength,
    #                                                       '\n pool_size1:', pollsize1, '\n pool_size2', pollsize2,
    #                                                       '\n dense_param:', denseprm, '\n epochs', epoch, 
    #                                                       '\n batch_size:', batchsz )
                                                    
                                                    cnn1dNet.fit(training_series, training_series_y,epochs=epoch, batch_size=batchsz, verbose=0)
                                                   
                                                    # evaluate the model
                                                    scores = cnn1dNet.evaluate(validation_series, validation_series_y, verbose=0)
                                                    #print("%s: %.2f%%" % (cnn1dNet.metrics_names[1], scores[1]*100))
                                                    cvscores.append(scores[1] * 100)
                                                    #print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
        
    #                                             print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
    #                                             print('\n\nWindow size:', ws, '\n optimizer:', opt, '\n activation: ', actv,
    #                                                       '\n nb_filter:', nbfltr, '\n filter_length:', fltrlength,
    #                                                       '\n pool_size1:', pollsize1, '\n pool_size2', pollsize2,
    #                                                       '\n dense_param:', denseprm, '\n epochs', epoch, 
    #                                                       '\n batch_size:', batchsz )
                                                
                                                xx=hyperTuningParameters(window_size=ws, optimizer=opt, activation=actv,
                                                                      nb_filter=nbfltr, filter_length=fltrlength,
                                                                      pool_size1 = pollsize1,pool_size2= pollsize2,
                                                                      dense_param=denseprm,epochs=epoch, batch_size=batchsz,
                                                                      scoreParams=numpy.mean(cvscores), stddevParams=numpy.std(cvscores))
                                                scoreList.append(xx)
                                                print("score:"+str(xx.scoreParams))
                                            except: # catch *all* exceptions
                                                e = sys.exc_info()[1]
                                                print( "<p>!!!!!!!!!Error: %s</p>" % e )
                                                
    with open(outfilename, 'w') as filexx:
        for scoreitem in scoreList:
            filexx.write("\n\nWS:"+ str(scoreitem.window_size) +" optimizer:"+ str(scoreitem.optimizer)
                              +" activation:"+str(scoreitem.activation) +" nb_filter:"+str(scoreitem.nb_filter)
                              +" filter_length:"+str(scoreitem.filter_length)+" pool_size1:" +str(scoreitem.pool_size1)  
                              +" pool_size2:"+str(scoreitem.pool_size2) + " dense_param:" +str(scoreitem.dense_param)
                              +" epochs:"+str(scoreitem.epochs) + " batch_size:"+str(scoreitem.batch_size)
                              +"\n Accuracy:" +str(scoreitem.scoreParams) + " Std_dev:"+str(scoreitem.stddevParams) +"\n")
        
            if(scoreitem.scoreParams>best_score):
                best_score=scoreitem.scoreParams
                best_score_parameters = scoreitem
                
                    
    print("WS:"+str(best_score_parameters.window_size)+" optimizer:"+ str(best_score_parameters.optimizer)+
          " activation:"+str(best_score_parameters.activation) +" nb_filter:"+str(best_score_parameters.nb_filter)
          +" filter_length:"+str(best_score_parameters.filter_length)+" pool_size1:" +str(best_score_parameters.pool_size1)  
          +" pool_size2:"+str(best_score_parameters.pool_size2) + " dense_param:" +str(best_score_parameters.dense_param)
          +" epochs:"+str(best_score_parameters.epochs) + " batch_size:"+str(best_score_parameters.batch_size)
          +" scoreParams:" +str(best_score_parameters.scoreParams) + " stddevParams:"+str(best_score_parameters.stddevParams))

def runResnet(parent_directory):
    "Prepare input data, build model, evaluate."
    np.set_printoptions(threshold=25)
   
    
    timeseries, timeseries_y = dataUtils.load_dataset(os.path.join(parent_directory, 'input/DataLearning_SpeechAct.h5'))
    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    
#     X, y, q = make_timeseries_instances(timeseries, timeseries_y, ws)
#     print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q)
#     print('\n\nInput shapes:', X.shape, '\n\nOutput shape:', y.shape, '\n\nQuery vector shape:', q.shape)
     
    
    # partition the data into training and testing splits using 90% of
    # the data for training and the remaining 10% for testing
    test_size = int(0.1 * nb_samples)           # In real life you'd want to use 0.2 - 0.5
    X_train_and_validation, X_test, y_train_and_validation, y_test = timeseries[:-test_size], timeseries[-test_size:], timeseries_y[:-test_size], timeseries_y[-test_size:] 
    nb_classes=2
    verbose=True
    import Colab_Resnet  
    
    if len(X_train_and_validation.shape) == 2: # if univariate 
        # add a dimension to make it multivariate with one dimension 
        X_train_and_validation = X_train_and_validation.reshape((X_train_and_validation.shape[0],X_train_and_validation.shape[1],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))

    input_shape = X_train_and_validation.shape[1:]
    
    ts_splits = TimeSeriesSplit(n_splits=5)
    cvscores = []
    indexNum=1;
    for train_index, validation_index in ts_splits.split(X_train_and_validation):
        if indexNum< 4:
            indexNum=indexNum+1
            continue
        training_series= X_train_and_validation[train_index]
        training_series_y= y_train_and_validation[train_index]
        
        validation_series= X_train_and_validation[validation_index]
        validation_series_y= y_train_and_validation[validation_index]

#                 fig, ax = plt.subplots()
#                 plot_cv_indices(ts_splits, X_train_and_validation, y_train_and_validation,  ax, n_splits)
        
        print('Observations: %d' % (len(training_series) + len(validation_series)))
        print('Training Observations x,y: %d, %d' %(len(training_series), len(training_series_y)) )
        print('Validation Observations x,y: %d, %d' % (len(validation_series),len(validation_series_y)))
                                       
        
        
        
        classifier=Colab_Resnet.Classifier_RESNET(parent_directory,input_shape, nb_classes, verbose)     
        scores = classifier.fit(training_series,training_series_y, validation_series,validation_series_y, validation_series_y)
        print("\n\n\n scoreParams:" +str(scores[1] * 100))
        #print("%s: %.2f%%" % (cnn1dNet.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        #print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

#                                             print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
#                                             print('\n\nWindow size:', ws, '\n optimizer:', opt, '\n activation: ', actv,
#                                                       '\n nb_filter:', nbfltr, '\n filter_length:', fltrlength,
#                                                       '\n pool_size1:', pollsize1, '\n pool_size2', pollsize2,
#                                                       '\n dense_param:', denseprm, '\n epochs', epoch, 
#                                                       '\n batch_size:', batchsz )
    
    print(" scoreParams:" +str(numpy.mean(cvscores)) + " stddevParams:"+str(numpy.std(cvscores)))


    

if __name__ == '__main__':

    print(sys.path)
    current_directory  = os.path.dirname(os.path.realpath('__file__'))
    parent_directory = os.path.split(current_directory)[0]
    
    #parent_directory="/content/drive/App/"
    parent_directory="./drive/My Drive/App/"
    outfilename = os.path.join(parent_directory, "output/scores2.txt")
    

    runResnet(parent_directory)
    #simpleCNN1d()
    
#     outpickle = os.path.join(parent_directory, "output/pickle.pkl")
#     f = open(outfilename, 'wb')
#     cpickle.dump(f, scoreList)
#     f.close()
#     with open(outfilename, 'w') as filexx:
#         for scoreitem in scoreList:
#             filexx.write("\n\nWS:"+ str(scoreitem.window_size) +" optimizer:"+ str(scoreitem.optimizer)
#                                   +" activation:"+str(scoreitem.activation) +" nb_filter:"+str(scoreitem.nb_filter)
#                                   +" filter_length:"+str(scoreitem.filter_length)+" pool_size1:" +str(scoreitem.pool_size1)  
#                                   +" pool_size2:"+str(scoreitem.pool_size2) + " dense_param:" +str(scoreitem.dense_param)
#                                   +" epochs:"+str(scoreitem.epochs) + " batch_size:"+str(scoreitem.batch_size)
#                                   +"\n Accuracy:" +str(scoreitem.scoreParams) + " Std_dev:"+str(scoreitem.stddevParams) +"\n")
#             
#             if(scoreitem.scoreParams>best_score):
#                 best_score=scoreitem.scoreParams
#                 best_score_parameters = scoreitem
#                 
#                     
#     print("WS:"+str(best_score_parameters.window_size)+" optimizer:"+ str(best_score_parameters.optimizer)+
#           " activation:"+str(best_score_parameters.activation) +" nb_filter:"+str(best_score_parameters.nb_filter)
#           +" filter_length:"+str(best_score_parameters.filter_length)+" pool_size1:" +str(best_score_parameters.pool_size1)  
#           +" pool_size2:"+str(best_score_parameters.pool_size2) + " dense_param:" +str(best_score_parameters.dense_param)
#           +" epochs:"+str(best_score_parameters.epochs) + " batch_size:"+str(best_score_parameters.batch_size)
#           +" scoreParams:" +str(best_score_parameters.scoreParams) + " stddevParams:"+str(best_score_parameters.stddevParams))
    
    
    
    
    
    
    