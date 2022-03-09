# -*- coding: utf-8 -*-

#https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0

import numpy as np
import h5py, os
from oauthlib.uri_validate import segment

class DataUtilsSA():
    
    def create_segments_and_labels(self, df, dflabels, time_steps, step):

        # x, y, z acceleration as features
        N_FEATURES = 20
        # Number of steps to advance in each iteration (for me, it should always
        # be equal to the time_steps in order to have no overlap between segments)
        # step = time_steps
        segments = []
        labels = []
        for i in range(0, len(df) - time_steps, step):
            npArr = np.array(df[i: i + time_steps])
#             npTransposedArr = npArr.transpose()
            label = dflabels[i + time_steps] #we just interested with the last gaze behavior
            segments.append(npArr)
            labels.append(label)
    
        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
        labels = np.asarray(labels)
    
        return reshaped_segments, labels

    
    def load_dataset(self, filename):
        "Load a gaze dataset"
        f = h5py.File(filename, 'r')
        X = f['X']
        Y = f['y']

        
        print (X.shape)
        print (Y.shape)
        return X,Y
        #l = len(X)
        #l_train = int(l*0.6)
        #l_valid = int(l*0.1)
        #l_test = int(l*0.3)
    
        # Split data into sets
        #X_train = X[1:l_train]
        #X_valid = X[l_train:(l_train+l_valid)]
        #X_test = X[(l_train+l_valid):(l_train+l_valid+l_test)]
    
        #print X_train.shape
    
        #y_train = Y[1:l_train]
        #y_valid = Y[l_train:(l_train+l_valid)]
        #y_test = Y[(l_train+l_valid):(l_train+l_valid+l_test)]
        #return X_train, y_train, X_test, y_test, X_valid, y_valid
    
    def plain_text_file_to_dataset(self, filename, outfilename, input_size, parent_directory):
        "Load a plain text file and construct a gaze dataset from it"""
        data=[]
        labels=[]
     
        
        #For accessing the file in a folder contained in the current folder
        filename = os.path.join(parent_directory, filename)
        indexnum=1;
        with open(filename, 'rb') as f:
            print ("Converting plain text file to trainable dataset (as h5file)")
            print ("Processing file " + filename + " as input")
            print ("input_size parameter (i.e. num of neurons) will be " + str(input_size))
            #columns SpeechRole SpeechActcolumns:(konusma konusmaesnasındaara dusunme konusmayabaslamadan micropause) InterviewerGender IntervieweeGender IntervieweeGaze
            for line in f:  
                if indexnum==1:
                    indexnum=indexnum+1
                    continue
                #l = len(line)
                #line = [int(s) for s in line.split() if s.isdigit()]
                line = [float(s) for s in line.split()]
                featureArray=line[:-2]
                
                gbFeature=line[-1]
                # Columns excluding the last are the inputs
                data.extend([np.append(featureArray,gbFeature)])
                # Take the last column to be the label
                label = line[-2]
                labels.extend([label])
                
            
    
            outfilename = os.path.join(parent_directory, outfilename)
                
#             with h5py.File(outfilename,'w') as H:
#                 print ("Shape of X: ", len(data), " x ", len(data[0]))
#                 print ("Writing data and labels to file " + outfilename)
#                 data = np.asarray(data)
#                 labels = np.asarray(labels)
#                 H.create_dataset( 'X', data=data ) # note the name X given to the dataset!
#                 H.create_dataset( 'y', data=labels ) # note the name y given to the dataset!
            TIME_PERIODS =9
            STEP_DISTANCE =3  
            with h5py.File(outfilename,'w') as H:
                 
                x_data, y_data = dataUtils.create_segments_and_labels(data, labels, time_steps=TIME_PERIODS, step=STEP_DISTANCE) 
                print ("Shape of X: ", x_data.shape)
                print ("Writing data and labels to file " + outfilename)
                data = np.asarray(x_data)
                labels = np.asarray(y_data)
                H.create_dataset( 'X', data=data ) # note the name X given to the dataset!
                H.create_dataset( 'y', data=labels ) # note the name y given to the dataset!  
    
          
             
    
    
    

dataUtils =DataUtilsSA()

if __name__ == '__main__':
    
    parent_directory="./drive/My Drive/App/"
    
    filename="input/TimeSeriesLearningData_NotEmptyGB_SpeechAct_OrderedByInterviewers_1_2_3_4_5_6_7.txt"
    dataUtils.plain_text_file_to_dataset(filename, outfilename='input/TimeSeriesLearningData_Interviewer_NotEmptyGB_SpeechAct_OrderedByInterviewers_1_2_3_4_5_6_7.h5', input_size=20, parent_directory=parent_directory)
    
    
    filename="input/TimeSeriesLearningData_NotEmptyGB_SpeechAct_OrderedByInterviewers_2_7_3_5_1_4_6.txt"
    dataUtils.plain_text_file_to_dataset(filename, outfilename='input/TimeSeriesLearningData_Interviewer_NotEmptyGB_SpeechAct_OrderedByInterviewers_2_7_3_5_1_4_6.h5', input_size=20, parent_directory=parent_directory)
    
    
    filename="input/TimeSeriesLearningData_NotEmptyGB_SpeechAct_OrderedByInterviewers_3_1_4_7_6_2_5.txt"
    dataUtils.plain_text_file_to_dataset(filename, outfilename='input/TimeSeriesLearningData_Interviewer_NotEmptyGB_SpeechAct_OrderedByInterviewers_3_1_4_7_6_2_5.h5', input_size=20, parent_directory=parent_directory)
    
    
    filename="input/TimeSeriesLearningData_NotEmptyGB_SpeechAct_OrderedByInterviewers_4_6_5_2_7_1_3.txt"
    dataUtils.plain_text_file_to_dataset(filename, outfilename='input/TimeSeriesLearningData_Interviewer_NotEmptyGB_SpeechAct_OrderedByInterviewers_4_6_5_2_7_1_3.h5', input_size=20, parent_directory=parent_directory)
    
    
    filename="input/TimeSeriesLearningData_NotEmptyGB_SpeechAct_OrderedByInterviewers_6_3_2_1_7_5_4.txt"
    dataUtils.plain_text_file_to_dataset(filename, outfilename='input/TimeSeriesLearningData_Interviewer_NotEmptyGB_SpeechAct_OrderedByInterviewers_6_3_2_1_7_5_4.h5', input_size=20, parent_directory=parent_directory)
    
    
    #parent_directory='/home/uaa/eclipse-workspace/GazeModels/'  #:localde test ederken
   
    
    
    
    
    
    
    
    #dataUtils.load_dataset('/home/uaa/eclipse-workspace/GazeModels/input/DataLearning_SpeechAct.h5') :localde test ederken
