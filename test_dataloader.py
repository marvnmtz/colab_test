# -*- coding: utf-8 -*-
import unittest

from lib.dataloader.mlDataloader import *


# this unit test validates that the dataloader and the kfold valitation are working as intended
class DataloaderTestCase(unittest.TestCase):
    def setUp(self):
        self.k = 10
        # load dataframe with computed features, test will ot work without
        self.df_features = pd.read_pickle('df_features.pkl')

        self.dl = mlDataloader('df_features.pkl')
        self.train_test_list_x, self.train_test_list_y = self.dl.crossValidation(self.k)
        
    def test_k_fold_num_of_steps(self):
        self.assertEqual(len(self.train_test_list_x), self.k)
        self.assertEqual(len(self.train_test_list_y), self.k)

    def test_num_of_datapoints(self):
        for i in range(self.k):
            self.assertEqual(len(self.train_test_list_x[i]['train']) + len(self.train_test_list_x[i]['test']), len(self.df_features['file_name']))
            self.assertEqual(len(self.train_test_list_y[i]['train']) + len(self.train_test_list_y[i]['test']), len(self.df_features['file_name']))
            
