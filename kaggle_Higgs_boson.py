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

def AMSScore(s,b): 
    return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
    

#==============================================================================
# read data
#==============================================================================

train_file = "D:/xinyu/Excel_DM/data/Kaggle_HiggsBoson/training.csv"
print("loading train file")
train_df = pd.read_csv(train_file, header=0)        # Load the train file into a dataframe
train_df = train_df.replace(-999,np.NaN)

#==============================================================================
# delete columns and use mean to replace missing values
#==============================================================================

train_df_drop =  train_df.drop(['EventId','Weight','Label'],axis=1)
for thecol in train_df_drop .columns:
    mean_x = train_df_drop[thecol].dropna().mean()
    
    #print mean_x
    
    if len(train_df_drop[thecol][train_df_drop[thecol].isnull()]) > 0:
        train_df_drop.loc[ (train_df_drop[thecol].isnull()), thecol] = mean_x
        
train_df['Label']=train_df['Label'].map({'s':1,'b':0}).astype(int)
x_data = train_df_drop.values
x_data = np.add(x_data, np.random.normal(0.0, 0.0001, x_data.shape))
y_data = train_df['Label'].values
w_data = train_df['Weight'].values
#==============================================================================
# select patial data for feature selection
#==============================================================================
np.random.seed(45)
# Random number for training/validation splitting
r =np.random.rand(x_data.shape[0])
pcnt = 1.0 # 1.0 for all the data
print 'select  data for feature selection'
# First 90% are training
x_data = x_data[r<pcnt]
y_data = y_data[r<pcnt]
w_data = w_data[r<pcnt]
#==============================================================================
# seprete data to train and validation
#==============================================================================
print 'Assigning data to numpy arrays.'
# First 90% are training
X_train,X_valid,Y_train,Y_valid,W_train,W_valid = \
train_test_split(x_data, y_data, w_data,test_size=0.1, random_state=42)

#==============================================================================
# PolynomialFeature and PCA
#==============================================================================

min_max_scaler = preprocessing.MinMaxScaler()

#skb =  SelectKBest(chi2, k=10)
#
poly = PolynomialFeatures(2,interaction_only=False,include_bias =False)
#
#skb2 = SelectKBest(chi2, k=30)

##
pca = PCA(n_components =30)

selection = SelectKBest(k=30)

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
#combined_features = FeatureUnion([("pca", pca), ("poly", poly)])

#feaPipeline = Pipeline([("MinMaxScaler",min_max_scaler),\
#                        ("selectKBest",skb),\
#                        ("PolynomialFeatures",poly),\
#                        ("selectKBest2",skb2)])

feaPipeline = Pipeline([("MinMaxScaler",min_max_scaler),("FeatureUnion",combined_features)])
                        
feaPipeline.fit(X_train,Y_train)
X_train = feaPipeline.transform(X_train)

#==============================================================================
# # Lirst 10% are validation
#==============================================================================

X_valid  = feaPipeline.transform(X_valid)
#==============================================================================
# training 
#==============================================================================
estr = GBC(n_estimators=50, max_depth=5,min_samples_leaf=200,max_features=10,verbose=1)
estr.fit(X_train,Y_train) 
#estr = SVC(C=1.0,probability=True)



#==============================================================================
# grid search
#http://scikit-learn.org/stable/auto_examples/grid_search_digits.html
#http://scikit-learn.org/stable/auto_examples/randomized_search.html
#==============================================================================
#estr = GBC(verbose=1)
#tuned_parameters = {'n_estimators': [10,50],
#                     'max_depth': [5,10],
#                     'min_samples_leaf': [100,200],
#                     'max_features':[10]}
#                  
#
#scores = ['precision', 'recall']
#
#for score in scores:
#    print("# Tuning hyper-parameters for %s" % score)
#    print()
#
#    clf = GridSearchCV(estr, tuned_parameters, cv=5, scoring=score)
#    clf.fit(X_train, Y_train)
#
#    print("Best parameters set found on development set:")
#    print()
#    print(clf.best_estimator_)
#    print()
#    print("Grid scores on development set:")
#    print()
#    for params, mean_score, scores in clf.grid_scores_:
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean_score, scores.std() / 2, params))
#    print()
#
#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = Y_valid, clf.predict(X_valid)
#    print(classification_report(y_true, y_pred))
#    print()
##==============================================================================
## check
##==============================================================================

# Get the probaility outpestrut from the trained method, using the 10% for testing
prob_predict_train = estr.predict_proba(X_train)[:,1]
prob_predict_valid = estr.predict_proba(X_valid)[:,1]
 
# Experience shows me that choosing the top 15% as signal gives a good AMS score.
# This can be optimized though!
pcut = np.percentile(prob_predict_train,85)
 
# This are the final signal and background predictions
Yhat_train = prob_predict_train > pcut
Yhat_valid = prob_predict_valid > pcut
 
# To calculate the AMS data, first get the true positives and true negatives
# Scale the weights according to the r cutoff.
TruePositive_train = W_train*(Y_train==1.0)*(1.0/0.9)
TrueNegative_train = W_train*(Y_train==0.0)*(1.0/0.9)
TruePositive_valid = W_valid*(Y_valid==1.0)*(1.0/0.1)
TrueNegative_valid = W_valid*(Y_valid==0.0)*(1.0/0.1)
 
# s and b for the training
s_train = sum ( TruePositive_train*(Yhat_train==1.0) )
b_train = sum ( TrueNegative_train*(Yhat_train==1.0) )
s_valid = sum ( TruePositive_valid*(Yhat_valid==1.0) )
b_valid = sum ( TrueNegative_valid*(Yhat_valid==1.0) )
 
# Now calculate the AMS scores
print 'Calculating AMS score for a probability cutoff pcut=',pcut

print '   - AMS based on 90% training   sample:',AMSScore(s_train,b_train)
print '   - AMS based on 10% validation sample:',AMSScore(s_valid,b_valid)

#
##==============================================================================
## # Now we load the testing data, storing the data (X) and index (I)
#print 'Loading testing data'
##==============================================================================
test_file =  "D:/xinyu/Excel_DM/data/Kaggle_HiggsBoson/test.csv"
#test_df = pd.read_csv(test_file, header=0,nrows=1000)        # Load the test file into a dataframe
test_df = pd.read_csv(test_file, header=0)  
### fill -999 by mean value
test_df = test_df.replace(-999,np.NaN)

for thecol in test_df.columns:
    mean_x = test_df[thecol].dropna().mean()
    
    print mean_x
    
    if len(test_df[thecol][test_df[thecol].isnull()]) > 0:
        test_df.loc[ (test_df[thecol].isnull()), thecol] = mean_x


test_data = test_df.drop('EventId',axis=1).values


X_test_old = test_data

#X_test_old = np.add(X_test_old, np.random.normal(0.0, 0.0001, X_test_old.shape))

X_test = feaPipeline.transform(X_test_old)
I_test = test_df['EventId'].values
 
# Get a vector of the probability predictions which will be used for the ranking
print 'Building predictions'
Predictions_test = estr.predict_proba(X_test)[:,1]
# Assign labels based the best pcut
Label_test = list(Predictions_test>pcut)
Predictions_test =list(Predictions_test)
 
# Now we get the CSV data, using the probability prediction in place of the ranking
print 'Organizing the prediction results'
resultlist = []
for x in range(len(I_test)):
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