# -*- coding: utf-8 -*-
"""
Created on Wed Jul 09 16:50:48 2014

@author: Liu
"""

# -*- coding: utf-8 -*-
#basic code
#http://dbaumgartel.wordpress.com/2014/06/15/the-kaggle-higgs-challenge-beat-the-benchmarks-with-scikit-learn/

#polynomialfeatures
#http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

# memory issue solved

#https://github.com/albahnsen/scikit-learn/commit/8b825bd0663156553dbfc5daecb82856ebc4bb06


# feature union 
#http://scikit-learn.org/stable/modules/pipeline.html
#http://scikit-learn.org/stable/auto_examples/feature_stacker.html#example-feature-stacker-py


import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
import math
import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import (GridSearchCV, RandomizedSearchCV,
                                 ParameterGrid, ParameterSampler)
from sklearn.metrics import classification_report
import pickle
#==============================================================================
# load pickle
#==============================================================================
outFile1 = open('est1.pkl','rb')
outFile2 = open('est2.pkl','rb')
outFile3 = open('est3.pkl','rb')
outFile4 = open('est4.pkl','rb')
feaPipeline = pickle.load(outFile1)
estr = pickle.load(outFile2)
pcut = pickle.load(outFile3)
means = pickle.load(outFile4)
outFile1.close()
outFile2.close()
outFile3.close()
outFile4.close()
##==============================================================================
## # Now we load the testing data, storing the data (X) and index (I)
#print 'Loading testing data'
##==============================================================================
test_file =  "D:/xinyu/Excel_DM/data/Kaggle_HiggsBoson/test.csv"
#test_df = pd.read_csv(test_file, header=0,nrows=1000)        # Load the test file into a dataframe
test_df = pd.read_csv(test_file, header=0)#,nrows=200000)  
### fill -999 by mean value
test_df = test_df.replace(-999,np.NaN)

for thecol in test_df.columns:
    if len(test_df[thecol][test_df[thecol].isnull()]) > 0:
        mean_x = means[thecol] #test_df[thecol].dropna().mean()
        print mean_x
        test_df.loc[ (test_df[thecol].isnull()), thecol] = mean_x
        
rowLen = len(test_df.index) # total number of rows
test_df0=test_df

##########
test_df = test_df0[:200000]
test_data = test_df.drop('EventId',axis=1).values
X_test_old = test_data
#X_test_old = np.add(X_test_old, np.random.normal(0.0, 0.0001, X_test_old.shape))
X_test = feaPipeline.transform(X_test_old)
I_test1 = test_df['EventId'].values
print 'Building predictions'
Predictions_test1 = estr.predict(X_test)
#Predictions_test1 = estr.predict_proba(X_test)[:,1]
#############
test_df = test_df0[200000:400000]
test_data = test_df.drop('EventId',axis=1).values
X_test_old = test_data
#X_test_old = np.add(X_test_old, np.random.normal(0.0, 0.0001, X_test_old.shape))
X_test = feaPipeline.transform(X_test_old)
I_test2 = test_df['EventId'].values
print 'Building predictions'
Predictions_test2 = estr.predict(X_test)
#Predictions_test2 = estr.predict_proba(X_test)[:,1]
#############
test_df = test_df0[400000:]
test_data = test_df.drop('EventId',axis=1).values
X_test_old = test_data
#X_test_old = np.add(X_test_old, np.random.normal(0.0, 0.0001, X_test_old.shape))
X_test = feaPipeline.transform(X_test_old)
I_test3 = test_df['EventId'].values
print 'Building predictions'
Predictions_test3 = estr.predict(X_test)
#Predictions_test3 = estr.predict_proba(X_test)[:,1]
# merge
Predictions_test = np.concatenate((Predictions_test1,Predictions_test2,Predictions_test3),axis=0)
I_test = np.concatenate((I_test1,I_test2,I_test3),axis=0)

# Assign labels based the best pcut
Label_test = list(Predictions_test>pcut)
Predictions_test =list(Predictions_test)
 
# Now we get the CSV data, using the probability prediction in place of the ranking
print 'Organizing the prediction results'
resultlist = []
for x in range(len(Predictions_test)):
    resultlist.append([int(I_test[x]), Predictions_test[x], 's'*(Label_test[x]==1.0)+'b'*(Label_test[x]==0.0)])
 
# Sort the result list by the probability prediction
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[1])
 
# Loop over result list and replace probability prediction with integer ranking
for y in range(len(resultlist)):
    resultlist[y][1]=y+1
 
# Re-sort the result list according to the index
resultlist = sorted(resultlist, key=lambda a_entry: a_entry[0])
 
#==============================================================================
# # Write the result list data to a csv file
#==============================================================================
print 'Writing a final csv file Kaggle_higgs_prediction_output.csv'
fcsv = open('D:/xinyu/Excel_DM/data/Kaggle_HiggsBoson/Kaggle_higgs_test_liu_wen.csv','w')
fcsv.write('EventId,RankOrder,Class\n')
for line in resultlist:
    theline = str(line[0])+','+str(line[1])+','+line[2]+'\n'
    fcsv.write(theline)
fcsv.close()
print "done"

#plt.figure(figsize=(8,4))
#
#plt.scatter(train_data[:,2],train_data[:,3],marker=">")
#
#plt.show()