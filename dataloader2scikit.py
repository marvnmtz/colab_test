# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, make_scorer, accuracy_score
from lib.dataloader.mlDataloader import *
import pickle
import pandas as pd

dl = mlDataloader('df_features_train_all.pkl', normalize='standart')

train_test_list_x, train_test_list_y, _ = dl.getData()
train_test_list_x_new = pd.DataFrame()

#%% Deleting columns
for feature in df_imp_total["feature"]:
    if df_imp_total["importance"][(df_imp_total["feature"]==feature)].values != 0:
        train_test_list_x_new = pd.concat((train_test_list_x_new, train_test_list_x[feature]), axis=1)

#%% PCA - reducing dimension of feature matrix
from sklearn.decomposition import PCA
pca = PCA(n_components=500)
principalComponents = pca.fit_transform(train_test_list_x_new)

train_test_list_x =  principalComponents

#%% Cross Validation
cov_detector = RandomForestClassifier(n_estimators = 100, random_state=1,criterion= 'entropy', class_weight = 'balanced')#max_depth=2
#cov_detector.fit(train_test_list_x[0]['train'].to_numpy(), train_test_list_y[0]['train'])

# pred = train_test_list_y[0]['test'] == cov_detector.predict(
#               train_test_list_x[0]['test'].to_numpy())
          

#print(cross_val_score(cov_detector, train_test_list_x.to_numpy(), train_test_list_y, scoring = make_scorer(balanced_accuracy_score), cv = 5))        
print(cross_val_score(cov_detector, train_test_list_x, train_test_list_y, scoring = make_scorer(balanced_accuracy_score), cv = 5))        


cov_detector.fit(train_test_list_x, train_test_list_y)
pred = cov_detector.predict(train_test_list_x)
filename = "_model.pkl"
with open(filename, 'wb') as file:
    pickle.dump(cov_detector, file)

print(balanced_accuracy_score(train_test_list_y, pred))