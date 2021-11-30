# -*- coding: utf-8 -*-
import os
import cv2
import random
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from multiprocessing import Pool
import timeit

from lib.preprocessing.compute_features import *

if __name__ == '__main__':
     
    colab = True  # Specify whether you run the script on google colab or not  
    traindata = True    
    
            
    df_features = pd.DataFrame()
    
    # load ground truth data
    if colab == True:
        start = "/content"
        sep = "/"
    else:
        start = os.getcwd()
        sep = "\\"
    path_txt = start + sep + "ISM"
    ground_truth = []
    if traindata == True:
        train_txt = open(path_txt + sep + "train.txt", "r")
        ground_truth = train_txt.readlines()
        train_txt.close()
        
    
    # load images
    if traindata == True:
        path_img = path_txt + sep + "train"
    else:
        path_img = path_txt + sep + "test"
        
    
    imglist = os.listdir(path_img)
    imglist = random.sample(imglist,100)  # Uncomment for testing
    
    start = timeit.default_timer()
    if colab == True:
        #%% Without Multiprocessing
        df_features = compute_features(path_img, imglist, ground_truth, traindata, colab)
    
    else:
        #%% With Multiprocessing
        # create sublists
        length_imglist = len(imglist)
        imglist1 = imglist[0:int(length_imglist/4)]
        imglist2 = imglist[int(length_imglist/4):int(length_imglist/2)]
        imglist3 = imglist[int(length_imglist/2):3*int(length_imglist/4)]
        imglist4 = imglist[3*int(length_imglist/4):length_imglist]
        
        # process data in parallel
        #df_features = compute_features(path_img, imglist, ground_truth, traindata)
        r = []
        with Pool(processes=4) as pool:
            r = pool.starmap(compute_features, [(path_img, imglist1, ground_truth, traindata, colab),(path_img, imglist2, ground_truth, traindata, colab),(path_img, imglist3, ground_truth, traindata, colab),(path_img, imglist4, ground_truth, traindata, colab)])
        #r = compute_features(path_img, imglist1, ground_truth)
        df_features = pd.concat(r, ignore_index=True)
    
    stop = timeit.default_timer()
    print('Time for feature extraction: ', stop - start)
    df_features.to_pickle("df_features.pkl")