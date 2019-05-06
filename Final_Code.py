#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:35:20 2019

@author: dipesh
"""

import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

#import the data:
df1 = pd.read_csv('labs.csv')
print(df1.shape)
df2 = pd.read_csv('examination.csv')
print(df2.shape)
df3 = pd.read_csv('questionnaire.csv')
print(df3.shape)
df4 = pd.read_csv('demographic.csv')
print(df4.shape)

#Feature selection: 
filter_exam = ['SEQN','BMXWT','BMXHT']
filter_ques = ['DIQ010'] #'DID040','RHQ162' : age when learn to have diabetes / answer when pregrancy
filter_demo = ['RIAGENDR','RIDAGEYR']
filter_labUR = [col for col in df1 if col.startswith('UR')]

#DataSet creation:
df_labUR = df1[filter_labUR]  #all urine analysis
df_exam = df2[filter_exam] #Weight-heigh, 
df_exam = df_exam.rename(columns = {'SEQN' : 'ID','BMXWT' : 'Weight','BMXHT' : 'Height'})
df_ques = df3[filter_ques] #Answer to the diabetes, plus same question during pregnancy
df_demo = df4[filter_demo] #Gender, age
df_demo = df_demo.rename(columns = {'RIAGENDR' : 'Gender','RIDAGEYR' : 'Age'})

#Feature cleaning,

#Feature processsing: NaN, Index, other
df_ques['DIQ010'].value_counts()
df_ques.loc[df_ques['DIQ010'] == 2.0, 'Diabetes'] = 0
df_ques.loc[df_ques['DIQ010'] == 1.0, 'Diabetes'] = 1 #diabetes
df_ques.loc[df_ques['DIQ010'] == 3.0, 'Diabetes'] = 1 #high risk concerdered as diabetes for a first study
df_ques.drop(['DIQ010'],axis=1, inplace = True);

df_ques.info()

df_labUR.info()

df_labUR.dropna(thresh=7000,axis=1, inplace=True) #Keep the feature that have at least half the data (gurai)
#df_labUR.drop('URXUTRI',axis=1,inplace=True)
for col in df_labUR:
    df_labUR[col].fillna(df_labUR[col].std(),inplace=True)

#filling:
for col in df_exam:
    df_exam[col].fillna(df_exam[col].mean(),inplace=True)
for col in df_demo:
    df_demo[col].fillna(df_demo[col].median(), inplace=True)

df_exam['Ratio']=df_exam['Weight']/df_exam['Height']

df = pd.concat([df_exam, df_demo], axis=1, join='inner')
df = pd.concat([df, df_labUR], axis=1, join='inner')
df = pd.concat([df, df_ques], axis=1, join='inner')

df.info()

#Drop some of the non encoded diabetes (didn't answer or so)
df['Diabetes'].fillna('NaN')
df.dropna(how='any',axis=0,inplace=True)


#df.drop(['ID'],axis=1,inplace=True)
df.drop(['ID','Age','Gender','Height','Weight'],axis=1,inplace=True)

df.shape

#Data for training
from sklearn.model_selection import train_test_split
y=df['Diabetes']
X=df.drop('Diabetes',axis=1)

from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(kind='borderline2').fit_sample(X, y)

YR=pd.Series(y_resampled)
XR=pd.DataFrame(X_resampled)

YR.shape
XR.shape

X_train, X_test, y_train, y_test = train_test_split(XR,YR,random_state=0)

X_train.shape

print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


#Model creation:


# 1. Catboost

from catboost import CatBoostClassifier

cb = CatBoostClassifier()

params_cb = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1, 4, 9],
         'iterations': [200],
         #'verbose': [True],
         #'loss_function':['Logloss']
         }


#model = CatBoostClassifier()

from sklearn.model_selection import GridSearchCV

#cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = 3)

grid_search_lb = GridSearchCV(cb, param_grid=params_cb, cv = 3, scoring="roc_auc", verbose=True)


grid_search_lb.fit(X_train, y_train)

grid_search_lb.best_params_



cb_model = CatBoostClassifier(iterations=200,
                              learning_rate=0.15,
                              l2_leaf_reg= 4,
                              depth=10,
                              verbose=True,
                              #eval_metric='auc'
                             )

cb_model.fit(X_train, y_train)

#Scoring
ycat_pre=cb_model.predict(X_test)
len(ycat_pre)

from sklearn.metrics import log_loss, f1_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc
#yrdn_pre=clf.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, cb_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

Result=pd.DataFrame()
Result['Test']=["Logloss","F1 Score","Precision","Accuracy", 'ROC AUC']
Result['Catboost']=[log_loss(y_test,ycat_pre),
                 f1_score(y_test,ycat_pre),
                 precision_score(y_test,ycat_pre),
                 accuracy_score(y_test,ycat_pre),
                 roc_auc
                ]
print(Result)
print(sns.heatmap(confusion_matrix(y_test,ycat_pre),annot=True,fmt="d",cbar=False,cmap='Blues'))

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: CatBoost')
plt.legend(loc="lower right")
plt.show()

#sns.heatmap(confusion_matrix(y_test,ycat_pre),annot=True,fmt="d",cbar=False,cmap='Blues')


# 2 KNN

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

from sklearn.metrics import log_loss, f1_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc
yknn_pre=knn.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, knn.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
Result['KNN']=[log_loss(y_test,yknn_pre),
                 f1_score(y_test,yknn_pre),
                 precision_score(y_test,yknn_pre),
                 accuracy_score(y_test,yknn_pre),
                 roc_auc
                ]
print(Result)
print('\n')
print(sns.heatmap(confusion_matrix(y_test,yknn_pre),annot=True,fmt="d",cbar=False,cmap='Blues'))

#sns.heatmap(confusion_matrix(y_test,yrdn_pre),annot=True,fmt="d",cbar=False,cmap='Blues')

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: KNN')
plt.legend(loc="lower right")
plt.show()

# NO Important features of KNN



#3 LightGBM

import lightgbm as lgb
  
lg = lgb.LGBMClassifier(silent=False)    
param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [100]
              }

grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)

grid_search.fit(X_train, y_train)
    
grid_search.best_estimator_
    
#d_train = lgb.Dataset(X_train, label=y_train)


light_gbm_model = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0, 
                                     importance_type='split', learning_rate=0.1, max_depth=50,
                                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0, 
                                     n_estimators=100, n_jobs=-1, num_leaves=300, objective=None, 
                                     random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=False, 
                                     subsample=1.0, subsample_for_bin=200000, subsample_freq=0)


#light_gbm_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',
#                       num_class=1,early_stopping = 50,num_iteration=10000,num_leaves=31,
 #                      is_enable_sparse='true',tree_learner='data',min_data_in_leaf=400,max_depth=8,
  #                     learning_rate=0.1, n_estimators=100, max_bin=255, subsample_for_bin=50000, 
   #                    min_split_gain=5, min_child_weight=5, min_child_samples=10, subsample=0.995, 
    #                   subsample_freq=1, colsample_bytree=1, reg_alpha=0, 
     #                  reg_lambda=0, seed=0, nthread=-1, silent=True)


light_gbm_model.fit(X_train, y_train, 
                    eval_set=[(X_test, y_test)], early_stopping_rounds=10, eval_metric='auc')


from sklearn.metrics import log_loss, f1_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc
ylight_pre=light_gbm_model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, light_gbm_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
Result['LightGBM']=[log_loss(y_test,ylight_pre),
                 f1_score(y_test,ylight_pre),
                 precision_score(y_test,ylight_pre),
                 accuracy_score(y_test,ylight_pre),
                 roc_auc
                ]
print(Result)
print('\n')
print(sns.heatmap(confusion_matrix(y_test,ylight_pre),annot=True,fmt="d",cbar=False,cmap='Blues'))

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: LightGBM')
plt.legend(loc="lower right")
plt.show()

# Important features of LightGBM

fea_imp = pd.DataFrame({'imp': light_gbm_model.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('Feature Importance: LightGBM')
plt.ylabel('Features')
plt.xlabel('Importance');


#4 SVM

#Train the Support Vector Classifier
from sklearn.svm import SVC

SVC_model = SVC(probability=True)

SVC_model.fit(X_train,y_train)

ySVC_pre=SVC_model.predict(X_test)

# Gridsearch

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(probability=True),param_grid,refit=True,verbose=3, cv=5)

grid.fit(X_train,y_train)

grid.best_params_

grid.best_estimator_

#grid_predictions = grid.predict(X_test)

from sklearn.metrics import log_loss, f1_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc
#ySVC_pre=SVC_model.predict(X_test)
grid_predictions = grid.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
Result['SVM']=[log_loss(y_test,grid_predictions),
                 f1_score(y_test,grid_predictions),
                 precision_score(y_test,grid_predictions),
                 accuracy_score(y_test,grid_predictions),
                 roc_auc
                ]
print(Result)
print('\n')
print(sns.heatmap(confusion_matrix(y_test,grid_predictions),annot=True,fmt="d",cbar=False,cmap='Blues'))

#sns.heatmap(confusion_matrix(y_test,yrdn_pre),annot=True,fmt="d",cbar=False,cmap='Blues')

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: SVM')
plt.legend(loc="lower right")
plt.show()



