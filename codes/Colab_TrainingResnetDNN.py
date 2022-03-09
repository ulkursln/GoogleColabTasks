
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
from Colab_TrainingResnet_DA_Interviewer import DAResnet32
from Colab_TrainingResnet_DA_Interviewer16 import  DAResnet16
from Colab_TrainingResnet_SpeechAct_Interviewer16 import SAResnet16
from Colab_TrainingResnet_SpeechAct_Interviewer32 import SAResnet32



if __name__ == '__main__':

#     print(sys.path)
#     current_directory  = os.path.dirname(os.path.realpath('__file__'))
#     parent_directory = os.path.split(current_directory)[0]
    
   
    parent_directory="./drive/My Drive/App/"
    #outfilename = os.path.join(parent_directory, "output/Resnet/scoresDA_Interviewer.txt")
    
    input_name= "input/TimeSeriesLearningData_Interviewer_NotEmptyGB_DA_OrderedByInterviewers_1_2_3_5_6_7.h5"
    daRsn32=DAResnet32()
    daRsn32.runResnet(parent_directory,input_name) 
    
    daRsn16=DAResnet16()
    daRsn16.runResnet(parent_directory,input_name) 
    
    input_name ="input/TimeSeriesLearningData_Interviewer_NotEmptyGB_SpeechAct_OrderedByInterviewers_1_2_3_4_5_6_7.h5"
    saRsn32=SAResnet32()
    saRsn32.runResnet(parent_directory,input_name)
    
    saRsn16=SAResnet16()
    saRsn16.runResnet(parent_directory,input_name) 
