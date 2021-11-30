from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import cv2
import random
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import numpy as np
from skimage import io, img_as_float
import pywt

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy

if __name__ ==  '__main__':
    #%% Load Images
    
    start = os.getcwd()
    print(start)
    
    # load ground truth data
    path_txt = start
    train_txt = open(path_txt + "//ISM//train.txt", "r")
    ground_truth = train_txt.readlines()
    train_txt.close()
    
    # load list of images paths
    path_img = start + "//ISM//train"
    imglist = os.listdir(path_img)
    
    
    n_samples = 1000  # 1000 for simplicity, later it should be changed to the number of images we have (including altered
    # images)
    images = []
    labels = []
    # Iteration through some pictures. Later it should go through all pictures
    for _ in range(n_samples):
        img_num = random.randint(0, len(imglist) - 1)  # Choosing a random number
        img = io.imread(path_img + "\\" + imglist[img_num])  # Load picture with that number
        img_float = img_as_float(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img_gray)  # List of all gray_scale pictures
        
        # Getting the output and converting it into a number
        diagnosis = ground_truth[img_num].split(" ")[1].strip()  # strip() to get rid of \n
        diagnosis_to_numeric = {"Normal": 0, "COVID": 1, "pneumonia": 2, "Lung_Opacity": 3}  # Converts string in numeric
        labels.append(diagnosis_to_numeric[diagnosis])
        
    #     
    images = np.array(images)
    labels = np.array(labels)
    
    #%% Feature Extraction with pooling
    import timeit
    from multiprocessing import Pool
    from feature_extractor_raika import *
    
    # Pooling
    start = timeit.default_timer()
    # create sublists
    length_imglist = len(images)
    imglist1 = images[0:int(length_imglist/4)]
    imglist2 = images[int(length_imglist/4):int(length_imglist/2)]
    imglist3 = images[int(length_imglist/2):3*int(length_imglist/4)]
    imglist4 = images[3*int(length_imglist/4):length_imglist]
    
    df_features = pd.DataFrame()  # DataFrame to store all the features 

    print("Performing simple statistics")
    with Pool(processes=4) as pool:
        # Simple statistics
        df = pd.DataFrame()
        r = []
        r = pool.map(simple_statistics, [(imglist1),(imglist2),(imglist3),(imglist4)])
        for i in range(len(r)):
            df = df.append(r[i])
        df_features = pd.concat([df_features,df], axis=1)
        '''
        # GLCM
        print("Performing GLCM")
        df = pd.DataFrame()
        r = []
        r = pool.map(GLCM_extractor, [(imglist1),(imglist2),(imglist3),(imglist4)])
        for i in range(len(r)):
            df = df.append(r[i])
        df_features = pd.concat([df_features,df], axis=1)
        '''
        # Gabor filters
        print("Performing Gabor filters")
        df = pd.DataFrame()
        r = []
        r = pool.map(gabor_extractor, [(imglist1),(imglist2),(imglist3),(imglist4)])
        for i in range(len(r)):
            df = df.append(r[i])
        df_features = pd.concat([df_features,df], axis=1)

        # LBP
        print("Performing LBP")
        df = pd.DataFrame()
        r = []
        r = pool.map(LBP_extractor, [(imglist1),(imglist2),(imglist3),(imglist4)])
        for i in range(len(r)):
            df = df.append(r[i])
        df_features = pd.concat([df_features,df], axis=1)
        
    stop = timeit.default_timer()
    print('Time Pooling: ', stop - start)  # For performance measuring
    #%% Without Pooling
    """start = timeit.default_timer()
    df_features = pd.DataFrame()  # DataFrame to store all the features 
         
    
    df_features = pd.concat([df_features,simple_statistics(images)], axis=1)
    df_features = pd.concat([df_features,GLCM_extractor(images)], axis=1)
    df_features = pd.concat([df_features,gabor_extractor(images)], axis=1)
    #df_features = pd.concat([df_features,wavelet_extractor(images)], axis=1)
    #df_features = pd.concat([df_features,entropy_extractor(img)], axis=1)
    
    stop = timeit.default_timer()
    print('Time Without Pooling: ', stop - start)
    """    
              
    #%% Scaling feature between 0 and 1
    result = df_features.copy()
    for feature_name in df_features.columns:
        max_value = result[feature_name].max()
        min_value = result[feature_name].min()
        if max_value != min_value:
            result[feature_name] = (result[feature_name] - min_value) / (max_value - min_value)
     
    features_scaled = result
    features_np = np.array(features_scaled)
    
    #%% PCA - reducing dimension of feature matrix
    from sklearn.decomposition import PCA
    pca = PCA(n_components=150)
    principalComponents = pca.fit_transform(features_np)
    
    
    #%% Splitting data into training and test data (x: images, y: labels)
    #X_train, X_test, y_train, y_test = train_test_split(features_np, labels, test_size=0.2, random_state=1234) 
    X_train, X_test, y_train, y_test = train_test_split(principalComponents, labels, test_size=0.2, random_state=1234)
    
    #%% Classification 
    clsf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=1, class_weight="balanced")
    clsf.fit(X_train, y_train)
    y_pred = clsf.predict(X_test)
    y_zero = y_pred * 0
    #Print overall accuracy
    from sklearn import metrics
    print ("Accuracy = ", metrics.accuracy_score(y_test, y_pred))
    print ("Accuracy Zeros = ", metrics.accuracy_score(y_test, y_zero))
    
    

    
    #search for corresponding entry in ground truth list
    #diagnosis = ""
    #for line in ground_truth:
    #    if line.split(" ")[0] == imglist[img_num]:
    #        diagnosis = line.split(" ")[1]
            
    # show image and diagnosis        
    #cv2.imshow('diagnosis: ' + diagnosis, img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()