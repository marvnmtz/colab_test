# -*- coding: utf-8 -*-
import cv2
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from scipy.stats import kurtosis, skew
import pywt
import numpy as np
from skimage.measure import shannon_entropy

def compute_features(path_img, imglist, ground_truth, traindata):
    data = {'file_name':[], 'diagnosis': [],'Mean': [],'Var': [], 'Kurtosis': [], 'Skewness': []
            , 'cH2_mean': [], 'cH2_var': [], 'cV2_mean': [], 'cV2_var': [], 'cD2_mean': [], 'cD2_var': []
            , 'energy': [], 'correlation': [], 'dissimilarity': [], 'homogeneity': [], 'contrast': []
            , 'energy2': [], 'correlation2': [], 'dissimilarity2': [], 'homogeneity2': [], 'contrast2': []
            , 'energy3': [], 'correlation3': [], 'dissimilarity3': [], 'homogeneity3': [], 'contrast3': []
            , 'energy4': [], 'correlation4': [], 'dissimilarity4': [], 'homogeneity4': [], 'contrast4': []
            , 'entropy': []}
    df = pd.DataFrame(data=data)
    for file_name in imglist:
        if file_name.split('.')[1] == 'png':
            try:
                # load file and convert to grayscale
                img = cv2.imread(path_img + "\\" + file_name)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # compute simple stats
                mean                                            = img.mean()
                var                                             = img.var()
                kur                                             = kurtosis(img, axis=None)
                skewness                                        = skew(img, axis=None)
                
                # compute wavelets
                
                C = pywt.wavedec2(img, 'db3', mode="periodization", level=2)
                #cA = C[0]
                (cH2, cV2, cD2) = C[1]
                (cH1, cV1, cD1) = C[2]
                #ca_arr = cA.reshape(-1)
                cH2_mean                                        = np.mean(cH2)
                cH2_var                                         = np.var(cH2) 
                cV2_mean                                        = np.mean(cV2)
                cV2_var                                         = np.var(cV2)
                cD2_mean                                        = np.mean(cD2)
                cD2_var                                         = np.var(cD2) 
                
                
                # compute greyscal 
                
               
                #angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                # props                                           = ['energy', 'correlation', 'dissimilarity', 'homogeneity', 'contrast']
                GLCM                                            = greycomatrix(img_gray, distances=[1], angles=[0])
        
                energy                                          = greycoprops(GLCM, prop='energy')[0][0]
                correlation                                     = greycoprops(GLCM, prop='correlation')[0][0]
                dissimilarity                                   = greycoprops(GLCM, prop='dissimilarity')[0][0]
                homogeneity                                     = greycoprops(GLCM, prop='homogeneity')[0][0]
                contrast                                        = greycoprops(GLCM, prop='contrast')[0][0]
                
                GLCM                                            = greycomatrix(img_gray, distances=[3], angles=[0])
        
                energy2                                          = greycoprops(GLCM, prop='energy')[0][0]
                correlation2                                     = greycoprops(GLCM, prop='correlation')[0][0]
                dissimilarity2                                   = greycoprops(GLCM, prop='dissimilarity')[0][0]
                homogeneity2                                     = greycoprops(GLCM, prop='homogeneity')[0][0]
                contrast2                                        = greycoprops(GLCM, prop='contrast')[0][0]
          
                GLCM                                            = greycomatrix(img_gray, distances=[5], angles=[0])
        
                energy3                                          = greycoprops(GLCM, prop='energy')[0][0]
                correlation3                                     = greycoprops(GLCM, prop='correlation')[0][0]
                dissimilarity3                                   = greycoprops(GLCM, prop='dissimilarity')[0][0]
                homogeneity3                                     = greycoprops(GLCM, prop='homogeneity')[0][0]
                contrast3                                        = greycoprops(GLCM, prop='contrast')[0][0]
                
                GLCM                                            = greycomatrix(img_gray, distances=[7], angles=[0])
        
                energy4                                          = greycoprops(GLCM, prop='energy')[0][0]
                correlation4                                     = greycoprops(GLCM, prop='correlation')[0][0]
                dissimilarity4                                   = greycoprops(GLCM, prop='dissimilarity')[0][0]
                homogeneity4                                    = greycoprops(GLCM, prop='homogeneity')[0][0]
                contrast4                                        = greycoprops(GLCM, prop='contrast')[0][0]
                # compute entropy
                entropy = shannon_entropy(img_gray)
                
                
                # add diagnosis
                diagnosis = ""
                if traindata:
                    for line in ground_truth:
                        if line.split(" ")[0] == file_name:
                            diagnosis = line.split(" ")[1]
                df = df.append({'file_name': file_name,'diagnosis': diagnosis,'Mean': mean,'Var': var, 'Kurtosis': kur, 'Skewness': skewness
                                , 'cH2_mean': cH2_mean, 'cH2_var': cH2_var, 'cV2_mean': cV2_mean, 'cV2_var': cV2_var, 'cD2_mean': cD2_mean, 'cD2_var': cD2_var
                                , 'energy': energy, 'correlation': correlation, 'dissimilarity': dissimilarity, 'homogeneity': homogeneity, 'contrast': contrast
                                , 'energy2': energy2, 'correlation2': correlation2, 'dissimilarity2': dissimilarity2, 'homogeneity2': homogeneity2, 'contrast2': contrast2
                                , 'energy3': energy3, 'correlation3': correlation3, 'dissimilarity3': dissimilarity3, 'homogeneity3': homogeneity3, 'contrast3': contrast3
                                , 'energy4': energy4, 'correlation4': correlation4, 'dissimilarity4': dissimilarity4, 'homogeneity4': homogeneity4, 'contrast4': contrast4
                                , 'entropy' : entropy}, ignore_index=True)   
            except:
                print("error:")
                print(file_name)
                
    return df