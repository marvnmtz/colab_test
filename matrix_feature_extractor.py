import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis

def matrix_features(matrix, label):
    # Sum over all elements
    sum_val = np.sum(matrix)
    
    density = np.mean(matrix)

    std_density = np.std(matrix)
    
    skewness = (skew(matrix, axis = None))
    
    kurt = (kurtosis(matrix, axis = None))
    
    energy = np.sum(np.square(matrix))

    value,counts = np.unique(matrix, return_counts=True)
    p = counts / np.sum(counts)
    p =  p[p!=0]
    entropy =-np.sum( p*np.log2(p))

    max_val = np.max(matrix)

    sum_deviation= np.sum(np.abs(matrix-np.mean(matrix)))
    mean_absolute_deviation = sum_deviation/len(matrix)

    median = np.median(matrix)

    min_val = np.min(matrix)

    range_val = max_val - min_val

    rms = np.sqrt(np.mean(np.square(matrix)))  # root_mean_square 

    uniformity = np.sum(np.square(p))
    
    # Creating List of Names for DataFrame
    feature_names = ["sum_val", "density", "std_density", "skew", "kurt", 
                     "energy", "entropy", "max", "MAD", "median", "min", 
                     "range", "RMS", "uniformity"]
    for idx, feature in enumerate(feature_names):
        feature_names[idx] = label + " " + feature

    # Creating Array containing all features
    features = np.array([sum_val, density, std_density,
        skewness, kurt, energy, entropy,
        max_val, mean_absolute_deviation, median, min_val, range_val, rms, uniformity])

    return features, feature_names
