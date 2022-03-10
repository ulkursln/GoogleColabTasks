# -*- coding: utf-8 -*-

from sklearn import preprocessing
import h5py
import os
import pandas as pd


class DataUtils():
    def load_dataset(self, filename):
        "Load a gaze dataset"
        f = h5py.File(filename, 'r')
        X = f['X']
        Y = f['y']

        
        print (X.shape)
        print (Y.shape)
        return X,Y
    
    def plain_text_file_to_dataset(self, filename, outfilename, input_size, parent_directory):
        "Load a plain text file and construct a gaze dataset from it"""
    
        filename = os.path.join(parent_directory, filename)
        df=pd.read_csv(filename, sep=" ")
        
        df_y= df[["InterviewerGB"]].copy()
        df_x_one_hot=df.drop(['InterviewerGB','IntervieweeGB', 'InterviewerGender', 'IntervieweeGender', 'functionalDependence', 'feedbackDependence' ], axis = 1) 
        
        df['IntervieweeIsAversion'] = df['IntervieweeGB'].map( {'Aversion':1, 'FaceContact':0} )
        df['InterviewerIsFemale'] = df['InterviewerGender'].map( {'F':1, 'M':0} )
        df['IntervieweeIsFemale'] = df['IntervieweeGender'].map( {'F':1, 'M':0} )
        df['IsfunctionalDependence'] = df['functionalDependence'].map( {'Yes':1, 'No':0} )
        df['IsfeedbackDependence'] = df['feedbackDependence'].map( {'Yes':1, 'No':0} )
        
        
        le = preprocessing.LabelEncoder()
        
        df_2 = df_x_one_hot.apply(le.fit_transform)
        df_2.head()
        
        enc = preprocessing.OneHotEncoder()
        enc.fit(df_2)
        
        onehot_x = enc.transform(df_2)
        df_x = pd.DataFrame(onehot_x.toarray())
        
        # add binary columns
        df_x['IntervieweeIsAversion'] = df['IntervieweeIsAversion']
        df_x['InterviewerIsFemale'] = df['InterviewerIsFemale']
        df_x['IntervieweeIsFemale'] = df['IntervieweeIsFemale']
        df_x['IsfunctionalDependence'] = df['IsfunctionalDependence'] 
        df_x['IsfeedbackDependence'] = df['IsfeedbackDependence'] 
        
        
        binary_y =pd.DataFrame()
        binary_y['InterviewerIsAversion']  =  df_y['InterviewerGB'].map( {'Aversion':1, 'FaceContact':0} )
        

        outfilename = os.path.join(parent_directory, outfilename)
 
        with h5py.File(outfilename,'w') as H:
            print ("Shape of X: ", df_x.shape)
            print ("Writing data and labels to file " + outfilename)
            H.create_dataset( 'X', data=df_x.values ) # note the name X given to the dataset!
            H.create_dataset( 'y', data=binary_y.values ) # note the name y given to the dataset!
                

    
dataUtils =DataUtils()

if __name__ == '__main__':
    filename="input/DALearningData.txt"

    dataUtils.plain_text_file_to_dataset(filename, outfilename='input/DataLearning_DA_Interviewer.h5', input_size=9, parent_directory='./drive/My Drive/App/')
    
#     current_directory  = os.path.dirname(os.path.realpath('__file__'))
#     parent_directory = os.path.split(current_directory)[0] # Repeat as needed
#     dataUtils.plain_text_file_to_dataset(filename, outfilename='input/DataLearning_DA_Interviewer.h5', input_size=9, parent_directory=parent_directory)
