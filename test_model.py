# -*- coding: utf-8 -*-
import pickle
from sklearn.ensemble import RandomForestClassifier
from lib.dataloader.mlDataloader import *

filename = "_model.pkl"

with open(filename, 'rb') as file:
    cov_detector = pickle.load(file)
    
dl = mlDataloader('df_features_test.pkl', normalize='standart')

train_test_list_x, train_test_list_y, files = dl.getData()

pred = cov_detector.predict(train_test_list_x)
print(pred)
file_list = files.tolist()
with open('pred.txt', 'w') as f:
    for index, file_name in enumerate(file_list):
        
        f.write(file_name + ' ' + pred[index])