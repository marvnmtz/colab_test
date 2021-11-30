# -*- coding: utf-8 -*-
import os
import random
import unittest
import pandas as pd

class FeatureExtractionTestCase(unittest.TestCase):
    def setUp(self):

        start = os.getcwd()
        # load dataframe with computed features, test will ot work without
        self.df_features = pd.read_pickle('df_features.pkl')
        self.randlist = []
        for i in range(200):
            self.randlist.append(random.randint(0, len(self.df_features)))
        # load ground truth data
        path_txt = start + "\\ISM"
        train_txt = open(path_txt + "\\train.txt", "r")
        self.ground_truth = train_txt.readlines()
        train_txt.close()
        
    def test_diagnosis(self):
         for index in self.randlist:
             name = self.df_features.loc[index]['file_name']
             for line in self.ground_truth:
                 if line.split(" ")[0] == name:
                     self.assertEqual(line.split(" ")[1], self.df_features.loc[index]['diagnosis'])
        
            
    def test_entries_unique(self):
        for index in self.randlist:
            name = self.df_features.loc[index]['file_name']
            for j in range( len(self.df_features)):
                if j != index:    
                    self.assertNotEqual(self.df_features.loc[j]['file_name'], name)