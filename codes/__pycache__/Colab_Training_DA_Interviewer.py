

#!/usr/bin/env python


from __future__ import print_function, absolute_import, division, unicode_literals, with_statement

from Colab_DataUtils_DA_Interviewer import DataUtils
from Colab_VGG import CNN1DModel as cnn1Dmodel
from sklearn.model_selection import TimeSeriesSplit
from matplotlib import pyplot

import os
import sys
import numpy as np
import pickle as cpickle


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

def make_timeseries_instances(timeseries, timeseries_y, window_size):
    
    timeseries = np.asarray(timeseries)
    assert 0 < window_size < timeseries.shape[0]
    X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    y = timeseries_y[window_size:]
    q = np.atleast_3d([timeseries[-window_size:]])
    return X, y, q



def train_CNN1d(parent_directory):

    "Prepare input data, build model, evaluate."
    np.set_printoptions(threshold=25)
   
    dataUtils =DataUtils()
    timeseries, timeseries_y = dataUtils.load_dataset(os.path.join(parent_directory, 'input/DataLearning_DA_Interviewer.h5'))
    nb_samples, nb_series = timeseries.shape
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
  

    batch_sizes = [16]
    epochs = [2,100]
    filter_numbers = [32]
    pool_size1 = [2]
    pool_size2 = [2]
    filter_length = [3]
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

    windows_size_arr=[13,8]
    for ws in windows_size_arr:
        X, y, q = make_timeseries_instances(timeseries, timeseries_y, ws)
        print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q)
        print('\n\nInput shapes:', X.shape, '\n\nOutput shape:', y.shape, '\n\nQuery vector shape:', q.shape)
         
        
      
        test_size = int(0.1 * nb_samples)         
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
                   
                                            cnn1Dmodel.window_size =ws
                                            try:
                                                
                                                cnn1dNet = cnn1Dmodel.create_model_VGG(nb_input_series=nb_series, nb_outputs=1,
                                                                          optimizer = opt,
                                                                          activation = actv, nb_filter=nbfltr,
                                                                          filter_length =fltrlength, pool_size1=pollsize1, 
                                                                          pool_size2=pollsize2, dense_param =denseprm)
            
                                                
                                                cnn1dNet.summary()

                                                
                                                ts_splits = TimeSeriesSplit(n_splits=5)

                                                cvscores = []
                                                for train_index, validation_index in ts_splits.split(X_train_and_validation):
                                                    training_series= X_train_and_validation[train_index]
                                                    training_series_y= y_train_and_validation[train_index]
                                                    
                                                    validation_series= X_train_and_validation[validation_index]
                                                    validation_series_y= y_train_and_validation[validation_index]
            

                                                    print('Observations: %d' % (len(training_series) + len(validation_series)))
                                                    print('Training Observations x,y: %d, %d' %(len(training_series), len(training_series_y)) )
                                                    print('Validation Observations x,y: %d, %d' % (len(validation_series),len(validation_series_y)))
 
                                                    hist=cnn1dNet.fit(training_series, training_series_y, epochs=epoch, batch_size=batchsz, validation_data=(validation_series, validation_series_y), verbose=1)
                                                    
                                                    scores = cnn1dNet.evaluate(X_test, y_test, verbose=0)
                                                    print("%s: %.2f%%" % (cnn1dNet.metrics_names[1], scores[1]*100))

                                      
                                                    
                                                    cvscores.append(scores[1] * 100)
                                                    with open(parent_directory +'output/VGG/Interviewer_index'+str(train_index)+'_score'+str(scores[1] * 100)+'_ws'+str(ws), 'wb') as file_hist:
                                                        cpickle.dump(hist.history, file_hist)
                                                   
                                                
                                                xx=hyperTuningParameters(window_size=ws, optimizer=opt, activation=actv,
                                                                      nb_filter=nbfltr, filter_length=fltrlength,
                                                                      pool_size1 = pollsize1,pool_size2= pollsize2,
                                                                      dense_param=denseprm,epochs=epoch, batch_size=batchsz,
                                                                      scoreParams=np.mean(cvscores), stddevParams=np.std(cvscores))
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

if __name__ == '__main__':


#     current_directory  = os.path.dirname(os.path.realpath('__file__'))
#     parent_directory = os.path.split(current_directory)[0] # Repeat as needed 
#     
#     with open('/home/uaa/eclipse-workspace/GazeModels/output/VGG/Interviewer_score94.84734493980875_ws13', 'rb') as f:
#         hist = cpickle.load(f,encoding='latin1')
# 
#     pyplot.plot(hist['binary_accuracy'], label='train')
#     pyplot.plot(hist['val_binary_accuracy'], label='validation')
#     #pyplot.plot(scores[1]*100, label ='test')
#     pyplot.legend()
#     pyplot.show()

    parent_directory="./drive/My Drive/App/"

    
    outfilename = os.path.join(parent_directory, "output/VGG/scoresDA_Interviewer.txt")
    

    
    train_CNN1d(parent_directory)
     

    f = open(outfilename, 'wb')
    cpickle.dump(f, scoreList)
    f.close()
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
        
        