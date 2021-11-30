# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:54:02 2021

@author: Marvi
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from lib.dataloader.mlDataloader import *
import pickle
import numpy as np
import pandas as pd
import os

start = os.getcwd()

dl = mlDataloader('df_features_train.pkl', normalize='standart')

train_test_list_x, train_test_list_y, _ = dl.getData()
#%% Calculate importance
model = ExtraTreesClassifier(n_estimators=100)
model.fit(train_test_list_x, train_test_list_y)
names = train_test_list_x.columns
importance = model.feature_importances_  # List of importance
imp = np.transpose(np.array([names, importance]))  # Array with names of features and importances

df_importance = pd.DataFrame(data=imp, columns=["feature","importance"])
df_importance_sorted = df_importance.sort_values(by=['importance'])

# Create rows containing the method (GLCM, Gabor etc.) and one containing the 
# statistics (mean, variance etc.)
d = []
for index, row in df_importance.iterrows():
    feature_split = row["feature"].split(" ")
    method = " ".join(feature_split[0:-1])
    statistic = feature_split[-1]
    d.append({'method': method, 'statistic': statistic})

df_importance_categorized = pd.DataFrame(d)

df_imp_total = pd.concat([df_importance, df_importance_categorized], axis=1)

# Calculate importance for each method and for each statistic
d = []
for method in pd.unique(df_imp_total["method"]):
    imp_only_method = df_imp_total[(df_imp_total["method"] == method)]["importance"]
    d.append({'method': method, 'sum': imp_only_method.sum(), 'average': imp_only_method.mean()})

df_methods = pd.DataFrame(d)



d = []
for statistic in pd.unique(df_imp_total["statistic"]):
    imp_only_statistic = df_imp_total[(df_imp_total["statistic"] == statistic)]["importance"]
    d.append({'statistic': statistic, 'sum': imp_only_statistic.sum(), 'average': imp_only_statistic.mean()})

df_statistic = pd.DataFrame(d)

#%% Save results to excel sheet 
df_methods_from_excel = pd.read_excel(start + "\\feature importance.xlsx", sheet_name="Method")#, index_col="IDX")
df_methods_from_excel = pd.concat((df_methods_from_excel, df_methods["average"]), axis=1)
df_statistic_from_excel = pd.read_excel(start + "\\feature importance.xlsx", sheet_name="Statistic")#, index_col="IDX")
df_statistic_from_excel = pd.concat((df_statistic_from_excel, df_statistic["average"]), axis=1)

with pd.ExcelWriter(start + "\\feature importance.xlsx") as writer:  
    df_methods_from_excel.to_excel(writer, sheet_name='Method', index=False)
    df_statistic_from_excel.to_excel(writer, sheet_name='Statistic', index=False)


    

    
