import pandas as pd
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import numpy as np
import pywt
import cv2
from scipy.stats import kurtosis, skew

def compute_features(matrix):
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    for img in images:
        new_row = pd.DataFrame()
        new_row["Mean"] = [img.mean()]
        new_row["Var"] = [img.var()]
        new_row["Kurtosis"] = [kurtosis(img, axis=None)]
        new_row["Skewness"] = [skew(img, axis=None)]
        df = df.append(new_row)
    
    return df 

def simple_statistics(images):
    """
    Calculat the statistics of the input image

    Parameters
    ----------
    img : Single gray-scale image, from which features get extracted.

    Returns
    -------
    df : pandas Data Frame containing the extracted features

    """
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    columns = ['Mean','Var', 'Kurtosis', 'Skewness']
    df = df.reindex(columns = df.columns.tolist() + columns)
    for img in images:
        new_row = pd.DataFrame(columns=columns)
        new_row["Mean"] = [img.mean()]
        new_row["Var"] = [img.var()]
        new_row["Kurtosis"] = [kurtosis(img, axis=None)]
        new_row["Skewness"] = [skew(img, axis=None)]
        df = df.append(new_row)
    
    return df

def GLCM_extractor(images):
    """
    GLCM is calculated for all combinations of px_distances and angles. 
    The the in props defined properties are calculated from the GLCM and stored
    in df.

    Parameters
    ----------
    images : List of gray-scale images, from which features get extracted

    Returns
    -------
    df : pandas Data Frame containing the extracted features

    """
    
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    px_distances = [1, 3, 5, 7]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    props = ['energy', 'correlation', 'dissimilarity', 'homogeneity', 'contrast']

    for img in images:                  # Iterate through all images
        new_row = pd.DataFrame()
        for px_dist in px_distances:    # Iterate through all distances
            for angle in angles:        # Iterate through all angles
                GLCM = greycomatrix(img, [px_dist], [angle]) 
                """Für alle Winkel die Matrix berechnen und dann den Mittelwert
                der 4 Rotationsmatrizen berechnen und daraus die Eigenschaften berechnen"""
                for prop in props:      # Iterate through all properties
                    name = prop + '_' + str(px_dist) + '_' + str(angle/np.pi) + 'pi'
                    new_row[name] = greycoprops(GLCM, prop)[0]
        
        df = df.append(new_row)  # After everything is calculated for this image
        # store the features in df

    return df
        
def wavelet_extractor(images):
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    
    for img in images:                  # Iterate through all images
        new_row = pd.DataFrame()
        """Wavelet transform""" 
        C = pywt.wavedec2(img, 'db3', mode="periodization", level=2)
        #cA = C[0]
        (cH2, cV2, cD2) = C[1]
        (cH1, cV1, cD1) = C[2]
        #ca_arr = cA.reshape(-1)
        new_row['cH2_mean'] = [np.mean(cH2)]
        new_row['cH2_var'] = [np.var(cH2)]  
        new_row['cV2_mean'] = [np.mean(cV2)]
        new_row['cV2_var'] = [np.var(cV2)] 
        new_row['cD2_mean'] = [np.mean(cD2)]
        new_row['cD2_var'] = [np.var(cD2)] 
        
        df = df.append(new_row)  # After everything is calculated for this image
        # store the features in df

    return df
    
def entropy_extractor(images):
    df = pd.DataFrame()
    for img in images:
        new_row = pd.DataFrame()
        entropy = shannon_entropy(img)
        new_row['Shannon Entropy'] = [entropy]
        df = df.append(new_row) # After everything is calculated for this image
        # store the features in df
        
    return df

def gabor_extractor(images):
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
    

    Parameters
    ----------
    images : List of gray-scale images, from which features get extracted

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = pd.DataFrame()

    ksize = 9  #Size of the kernel. Large kernels may not get small features
    sigma_list = (1, 3) #Large sigma on small features will fully miss the features. 
    theta_list = np.arange(0, np.pi, np.pi / 4)  #/4 shows horizontal 3/4 shows other horizontal. Try 
    # other contributions
    lamda_list = np.arange(0, np.pi, np.pi / 4)  #1/4 works best for angled. 
    gamma_list = (0.05, 0.5)  #Value of 1 defines spherical. Calue close to 0 has high aspect 
    # ratio. Value of 1, spherical may not be ideal as it picks up features 
    # from other regions.
    phi = 0  #Phase offset. I leave it to 0. 
    
    #Generate Gabor features
    for img in images:
        new_row = pd.DataFrame()
        num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
        for theta in theta_list:   #Define number of thetas. Here only 2 theta values 0 and 1/4 . pi 
            for sigma in sigma_list:  #Sigma with values of 1 and 3
                for lamda in lamda_list:   #Range of wavelengths
                    for gamma in gamma_list:   #Gamma values of 0.05 and 0.5
                                   
                        gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc. 
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)    
                        #Now filter the image and add values to a new column 
                        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel) # Filtering
                        gabor_labels = [gabor_label + " Mean", gabor_label + " Var", gabor_label + " Kurtosis",  gabor_label + " Skewness"]
                        feature_values = [fimg.mean(), 
                                          fimg.var(), 
                                          kurtosis(fimg, axis=None), 
                                          skew(fimg, axis=None)]
                        new_row[gabor_labels] =[feature_values]
    
                        num += 1  #Increment for gabor column label
                        
        df = df.append(new_row) # After everything is calculated for this image
        # store the features in df
        
    return df
    
    