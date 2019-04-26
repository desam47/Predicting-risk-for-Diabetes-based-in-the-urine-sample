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

#Model creation:

# 1. RandomForest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=15, max_depth=None, min_samples_split=4, random_state=0)

#Training
clf = clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_test, y_test)

from sklearn.metrics import log_loss, f1_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc
yrdn_pre=clf.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)

Result=pd.DataFrame()
Result['Test']=["Logloss","F1 Score","Precision","Accuracy", 'ROC AUC']
Result['Random F']=[log_loss(y_test,yrdn_pre),
                 f1_score(y_test,yrdn_pre),
                 precision_score(y_test,yrdn_pre),
                 accuracy_score(y_test,yrdn_pre),
                 roc_auc
                ]
print(Result)
print(sns.heatmap(confusion_matrix(y_test,yrdn_pre),annot=True,fmt="d",cbar=False,cmap='Blues'))

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

imp=clf.feature_importances_
feature=X.columns
tableFEAT=pd.DataFrame()
tableFEAT['Feature Name']=feature
tableFEAT['Score']=imp
tableFEAT.sort_values('Score',ascending=False,inplace=True)
print("Random Forest:")
tableFEAT


# 2. Catboost

from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=1500,
                           learning_rate=0.02,
                           depth=2,
                           loss_function='Logloss',
                           #od_wait=50,
                           verbose=True,
                          )

model.fit(X_train, y_train)

#Scoring
ycat_pre=model.predict(X_test)
len(ycat_pre)

from sklearn.metrics import log_loss, f1_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc
#yrdn_pre=clf.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
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
plt.title('ROC Curve: CatBoost ')
plt.legend(loc="lower right")
plt.show()

#sns.heatmap(confusion_matrix(y_test,ycat_pre),annot=True,fmt="d",cbar=False,cmap='Blues')

# Important features of catboost

fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('Feature Importance: CatBoost')
plt.ylabel('Features')
plt.xlabel('Importance');



# 3 KNN

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



#4 LightGBM

import lightgbm as lgb

 # Converting the training data into LightGBM dataset format
 #    sc = StandardScaler()
  #   X_train = sc.fit_transform(X_train)
   #  X_test = sc.transform(X_test)
    # d_train = lgb.Dataset(X_train, label=y_train)

#parameters = {
#     'scale_pos_weight': 1000,
#    'application': 'binary',
 #   'objective': 'binary',
  #  'metric': 'auc',
   # 'is_unbalance': 'true',
    #'boosting': 'gbdt',
    #'num_leaves': 31,
#    'feature_fraction': 0.5,
 #   'bagging_fraction': 0.5,
  #  'bagging_freq': 20,
   # 'learning_rate': 0.0009,
    #'verbose': 100,
    #'n_estimators': 150
#}

#light_gbm_model = lgb.LGBMClassifier(**parameters)

light_gbm_model = lgb.LGBMClassifier(boosting_type='gbdt', objective='binary',
                       num_class=1,early_stopping = 50,num_iteration=10000,num_leaves=31,
                       is_enable_sparse='true',tree_learner='data',min_data_in_leaf=400,max_depth=8,
                       learning_rate=0.1, n_estimators=100, max_bin=255, subsample_for_bin=50000, 
                       min_split_gain=5, min_child_weight=5, min_child_samples=10, subsample=0.995, 
                       subsample_freq=1, colsample_bytree=1, reg_alpha=0, 
                       reg_lambda=0, seed=0, nthread=-1, silent=True)


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

fig, ax = plt.subplots(figsize=(12, 8))
lgb.plot_importance(light_gbm_model, max_num_features=30, ax=ax)
plt.title("LightGBM - Feature Importance");


# Important features of LightGBM

fea_imp = pd.DataFrame({'imp': light_gbm_model.feature_importances_, 'col': X.columns})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
plt.title('LightGBM - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance');


# SVM

#Train the Support Vector Classifier
from sklearn.svm import SVC

SVC_model = SVC(probability=True)

SVC_model.fit(X_train,y_train)

ySVC_pre=SVC_model.predict(X_test)

from sklearn.metrics import log_loss, f1_score, precision_score, accuracy_score, confusion_matrix, roc_curve, auc
ySVC_pre=SVC_model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, SVC_model.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
Result['SVM']=[log_loss(y_test,ySVC_pre),
                 f1_score(y_test,ySVC_pre),
                 precision_score(y_test,ySVC_pre),
                 accuracy_score(y_test,ySVC_pre),
                 roc_auc
                ]
print(Result)
print('\n')
print(sns.heatmap(confusion_matrix(y_test,ySVC_pre),annot=True,fmt="d",cbar=False,cmap='Blues'))

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
