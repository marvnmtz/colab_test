# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class mlDataloader:
    def __init__(self, df_filename, normalize = None):

        df_features = pd.read_pickle(df_filename)
        self.y = df_features['diagnosis']
        self.filenames = df_features['file_name']
        df_features.drop(columns=['file_name', 'diagnosis'], inplace=True)
        if normalize == 'standart':
            for feature_name in df_features.columns: 
                max_value = float(df_features[feature_name].max())
                min_value = float(df_features[feature_name].min())
                if max_value != min_value:
                    df_features[feature_name] = (df_features[feature_name].astype(float) - min_value) / (max_value - min_value)
                    
        
        self.x = df_features
        
        #print(self.x)
        #print(self.y)
        
    def getData(self):
        return self.x, self.y, self.filenames
        
    def crossValidation(self, num_splits):
        kf = KFold(n_splits=num_splits, random_state=42, shuffle=True)
        index_list = []
        train_test_list_x = []
        train_test_list_y = []
        for train_index, test_index in kf.split(self.x):
            index_list.append([train_index, test_index])
            train_test_list_x.append({'train' : self.x.iloc[train_index], 'test' : self.x.iloc[test_index]})
            train_test_list_y.append({'train' : self.y[train_index], 'test' : self.y[test_index]})
        return train_test_list_x, train_test_list_y, self.filenames