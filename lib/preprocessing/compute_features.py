# -*- coding: utf-8 -*-
import cv2
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage.feature import local_binary_pattern
from scipy.stats import kurtosis, skew
from matrix_feature_extractor import matrix_features
import pywt
import numpy as np
from skimage.measure import shannon_entropy

def compute_features(path_img, imglist, ground_truth, traindata, colab):
    counter = 1
    FIRST_IMG = True
    sep = "/" if colab else "\\"  # Seperator depends on system
    for file_name in imglist:
        if counter%50 == 0:
            print(counter, ":", file_name)
        counter += 1
        if file_name.split('.')[1] == 'png':
            try:
                # load file and convert to grayscale
                img = cv2.imread(path_img + sep + file_name)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                #%% add filename and diagnosis to feature and label array
                diagnosis = ""
                if traindata:
                    for line in ground_truth:
                        if line.split(" ")[0] == file_name:
                            diagnosis = line.split(" ")[1]
                features = [file_name, diagnosis]
                feature_labels = ['file_name', 'diagnosis']
                
                #%% compute simple stats features
                features_this, feature_labels_this = matrix_features(img_gray, "Original")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)
                
                #%% compute wavelet features
                C = pywt.wavedec2(img, 'db3', mode="periodization", level=2)
                
                cA = C[0]
                (cH2, cV2, cD2) = C[1]
                (cH1, cV1, cD1) = C[2]
                
                features_this, feature_labels_this = matrix_features(cA, "cA")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)
                features_this, feature_labels_this = matrix_features(cH2, "cH2")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)
                features_this, feature_labels_this = matrix_features(cV2, "cV2")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)
                features_this, feature_labels_this = matrix_features(cD2, "cD2")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)
                features_this, feature_labels_this = matrix_features(cH1, "cH1")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)
                features_this, feature_labels_this = matrix_features(cV1, "cV1")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)
                features_this, feature_labels_this = matrix_features(cD1, "cD1")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)
                
                #%% compute GLCM features
                # Here the features from matrix_features are calculated as well
                # as some specific features for GLCMs
                
               
                px_distances = [1, 3, 5, 7]
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                props = ['energy', 'correlation', 'dissimilarity', 'homogeneity', 'contrast']

                for px_dist in px_distances:    # Iterate through all distances
                    GLCM = []
                    for angle in angles:        # Iterate through all angles
                        GLCM.append(greycomatrix(img_gray, [px_dist], [angle]))     
                    mean_GLCM = np.mean(GLCM, axis = 0) # Mean Matrix over all angles
                    label = "GLCM " + str(px_dist) + "px"
                    features_this, feature_labels_this = matrix_features(mean_GLCM, label)
                    features.extend(features_this)
                    feature_labels.extend(feature_labels_this)
                    for prop in props:      # Iterate through all properties
                        features.extend(greycoprops(mean_GLCM, prop)[0])  
                        feature_labels.extend([label + " " + prop])
                        
                     
                # compute entropy
                #entropy = shannon_entropy(img_gray)
                
                #%% compute Gabor features
                """
                The impulse response of a Gabor filter is defined by a sinusoidal wave (a 
                plane wave for 2-D Gabor filters) multiplied by a Gaussian function. 
                λ\lambda represents the wavelength of the sinusoidal factor, 
                θ\theta represents the orientation of the normal to the parallel stripes of 
                    a Gabor function, 
                ψ\psi is the phase offset, 
                σ\sigma is the sigma/standard deviation of the Gaussian envelope and 
                γ\gamma is the spatial aspect ratio, and specifies the ellipticity of the 
                    support of the Gabor function.

                """
                #"""
                ksize = 9  #Size of the kernel. Large kernels may not get small features
                sigma_list = (1, 3) #Large sigma on small features will fully miss the features. 
                theta_list = np.arange(0, np.pi, np.pi / 4)  #/4 shows horizontal 3/4 shows other horizontal. Try 
                # other contributions
                lamda_list = np.arange(np.pi / 4, np.pi, np.pi / 4)  #1/4 works best for angled. 
                gamma_list = (0.05, 0.5)  #Value of 1 defines spherical. Value close to 0 has high aspect 
                # ratio. Value of 1, spherical may not be ideal as it picks up features 
                # from other regions.
                phi = 0  #Phase offset. I leave it to 0.
                
                #Generate Gabor features

                num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
                for theta in theta_list:   #Define number of thetas. Here only 2 theta values 0 and 1/4 . pi 
                    for sigma in sigma_list:  #Sigma with values of 1 and 3
                        for lamda in lamda_list:   #Range of wavelengths
                            for gamma in gamma_list:   #Gamma values of 0.05 and 0.5
                                           
                                label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc. 
                                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)    
                                #Now filter the image and add values to a new column 
                                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel) # Filtering
                                features_this, feature_labels_this = matrix_features(fimg, label)
                                features.extend(features_this)
                                feature_labels.extend(feature_labels_this)
                                #print(num, theta, sigma, lamda, gamma)
            
                                num += 1  #Increment for gabor column label
                                    
                #"""                
                #%% compute LBP features
                LBP = local_binary_pattern(img_gray, 8, 2, method = 'uniform') # hier evtl nochmal Werte anpassen oder schleife über verschiedene Werte bauen
                features_this, feature_labels_this = matrix_features(LBP, "LBP")
                features.extend(features_this)
                feature_labels.extend(feature_labels_this)

                    
                
                
                #%% Create one array containing all the features
                if FIRST_IMG:
                    features_all = np.empty((0,len(features)))
                    FIRST_IMG = False
                features_all = np.vstack([features_all, features])

            except:
                print("error:")
                print(file_name)
                
    df = pd.DataFrame()
    df[feature_labels] = features_all
    
    return df