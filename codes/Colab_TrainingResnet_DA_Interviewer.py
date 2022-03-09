
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from DataUtils_DA_Interviewer import DataUtilsDA
from sklearn.model_selection import TimeSeriesSplit

import os
import numpy as np
import pickle as cpickle
import Colab_Resnet  



class DAResnet32():
    def runResnet(self, parent_directory, input_name):
        "Prepare input data, build model, evaluate."
        np.set_printoptions(threshold=25)
        n_feature_maps =32 #16
        batch_size = 64
        nb_epochs = 100
         
        dataUtilsDA =DataUtilsDA()
        timeseries, timeseries_y = dataUtilsDA.load_dataset(os.path.join(parent_directory, input_name))
        
        input_shape=(timeseries.shape[1],timeseries.shape[2])
        timeseriesArr=np.array(timeseries)
        nb_samples=timeseries.shape[0]
    
        #timeseriesArr = timeseriesArr.reshape(nb_samples, input_shape)
    
        #print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_timeStepsdotchannels), timeseries)
        
    
        test_size = int(0.1 * nb_samples)           
        X_train_and_validation, X_test, y_train_and_validation, y_test = timeseriesArr[:-test_size], timeseriesArr[-test_size:], timeseries_y[:-test_size], timeseries_y[-test_size:] 
     
      
        
        
        if len(X_train_and_validation.shape) == 2: # if univariate 
            # add a dimension to make it multivariate with one dimension 
            X_train_and_validation = X_train_and_validation.reshape((X_train_and_validation.shape[0],X_train_and_validation.shape[1],1))
            X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    
        
        ts_splits = TimeSeriesSplit(n_splits=5)
        cvscores = []
        
        indexNo=1
        for train_index, validation_index in ts_splits.split(X_train_and_validation):
            
            training_series= X_train_and_validation[train_index]
            training_series_y= y_train_and_validation[train_index]
            
            validation_series= X_train_and_validation[validation_index]
            validation_series_y= y_train_and_validation[validation_index]
    
    
            print('Observations: %d' % (len(training_series) + len(validation_series)))
            print('Training Observations x,y: %d, %d' %(len(training_series), len(training_series_y)) )
            print('Validation Observations x,y: %d, %d' % (len(validation_series),len(validation_series_y)))
                                           
           
            classifier=Colab_Resnet.Classifier_RESNET(parent_directory,input_shape, n_feature_maps, parent_directory +'output/Resnet/DA/')     
            
            
            mini_batch_size = int(min(training_series.shape[0]/10, batch_size))
    
            hist = classifier.model.fit(training_series, training_series_y, batch_size=mini_batch_size, epochs=nb_epochs,
                validation_data=(validation_series, validation_series_y), verbose=1, callbacks=classifier.callbacks)
                # evaluate the model
            scores = classifier.model.evaluate(validation_series, validation_series_y, verbose=0)
            
            
                                                        
            print("\n\n\n scoreParams:" +str(scores[1] * 100))
            #print("%s: %.2f%%" % (cnn1dNet.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            
            with open(parent_directory +'output/Resnet/DA/Interviewer_index'+str(indexNo)+'_score'+str(scores[1] * 100)+'_n_feature_maps'+str(n_feature_maps)+'_','wb') as file_hist:
                cpickle.dump(hist.history, file_hist)
           
            indexNo = indexNo+1
            
            
        model_json = classifier.model.to_json()
        with open(parent_directory +'output/Resnet/DA/InterviewerModel_nb_filter'+str(n_feature_maps)+'_.json', "w") as json_file:
            json_file.write(model_json)
        classifier.model.save_weights(parent_directory +'output/Resnet/DA/InterviewerModel_weights'+str(n_feature_maps)+'_.h5')
        
        with open(parent_directory +'output/Resnet/DA/InterviewerModelCVSAvrg_nb_filter'+str(n_feature_maps)+'_score'+str(np.mean(cvscores))+'_std'+ str(np.std(cvscores))+'.json', "w") as json_file:
            json_file.write(model_json)
        print(" scoreParams:" +str(np.mean(cvscores)) + " stddevParams:"+str(np.std(cvscores)))
    
        #keras.backend.clear_session()

    def runResnetforDifferentSetsAfterCV(self,parent_directory, input_name, output_drctr):
        "Prepare input data, build model, evaluate."
        np.set_printoptions(threshold=25)
        n_feature_maps = 32
        batch_size = 64
        nb_epochs = 100
         
        dataUtilsDA =DataUtilsDA()
        timeseries, timeseries_y = dataUtilsDA.load_dataset(os.path.join(parent_directory, input_name))
        
        input_shape=(timeseries.shape[1],timeseries.shape[2])
        timeseriesArr=np.array(timeseries)
        nb_samples=timeseries.shape[0]
        
    
        test_size = int(0.1 * nb_samples)           # In real life you'd want to use 0.2 - 0.5
        X_train, X_test, y_train, y_test = timeseriesArr[:-test_size], timeseriesArr[-test_size:], timeseries_y[:-test_size], timeseries_y[-test_size:] 
     
      
        
        
        if len(X_train.shape) == 2: # if univariate 
            # add a dimension to make it multivariate with one dimension 
            X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
            X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    
        

        cvscores = []
        
        

        print('Training Observations x,y: %d, %d' %(len(X_train), len(y_train)) )
        print('Validation Observations x,y: %d, %d' % (len(X_test),len(y_test)))
                                           
           
        classifier=Colab_Resnet.Classifier_RESNET(parent_directory,input_shape, n_feature_maps, output_drctr)     
            
            
        mini_batch_size = int(min(X_train.shape[0]/10, batch_size))
    
        hist = classifier.model.fit(X_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
        validation_data=(X_test, y_test), verbose=1, callbacks=classifier.callbacks)
                # evaluate the model
        scores = classifier.model.evaluate(X_test, y_test, verbose=0)
            
            
                                                        
        print("\n\n\n scoreParams:" +str(scores[1] * 100))
            #print("%s: %.2f%%" % (cnn1dNet.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
            
        with open(output_drctr+'_Valscore'+str(scores[1] * 100)+'_n_feature_maps'+str(n_feature_maps)+'_DA', 'wb') as file_hist:
            cpickle.dump(hist.history, file_hist)
           
            
        model_json = classifier.model.to_json()
        with open(output_drctr+'_'+str(n_feature_maps)+'_DA.json', "w") as json_file:
            json_file.write(model_json)
        classifier.model.save_weights(output_drctr+'_InterviewerModel_weights'+str(n_feature_maps)+'_DA.h5')
        
   
daresnet_32=DAResnet32()
if __name__ == '__main__':

#     print(sys.path)
#     current_directory  = os.path.dirname(os.path.realpath('__file__'))
#     parent_directory = os.path.split(current_directory)[0]
    
   
    parent_directory="./drive/My Drive/App/"
    
    #For hypertuning with bachtestfolding 
#     input_name= "input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_1_2_3_5_6_7.h5"
#     daresnet_32.runResnet(parent_directory, input_name)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_1_2_3_5_6_7.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/1/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_2_3_5_6_7_1.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/2/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_3_6_5_7_1_2.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/3/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_5_3_1_7_2_6.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/4/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_6_7_1_2_3_5.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/5/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_1_5_3_6_2_7.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/6/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_6_2_3_7_5_1.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/7/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_2_1_7_3_6_5.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/8/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_5_7_6_1_3_2.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/9/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_3_1_7_2_5_6.h5"
    output_fldr=parent_directory +'output/Resnet/DA/DifferentDataSets/10/'
    daresnet_32.runResnetforDifferentSetsAfterCV(parent_directory,input_name, output_fldr)
#     f = open(outfilename, 'wb')
#     cpickle.dump(f, scoreList)
#     f.close()
