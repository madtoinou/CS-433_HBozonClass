# -*- coding: utf-8 -*-
"""Implemented functions for preprocessing steps"""

import numpy as np

def clean_data(tX):
    '''Set NaN values to the value
       of the mean for each feature'''
    
    tX[tX <= -999] = np.nan
    col_mean = np.nanmean(tX, axis=0)
    inds = np.where(np.isnan(tX))
    tX[inds] = np.take(col_mean, inds[1])
    
    return tX


def log_transform(tX):
    """Perform logarithmic function on data, 
       useful for skewed distributions"""
    
    tX_logs = np.zeros((tX.shape))
    for i in range(tX.shape[1]):
        if np.all(tX[:,i] > 0):
            print(features_names[i])
            tX_log = np.log(tX[:,i])
            tX_logs[:,i] = tX_log
            plt.hist(tX_log, bins=2)
            plt.show()
        else:
            tX_log = tX[:,i]
            tX_logs[:,i] = tX_log
            plt.hist(tX_log, bins=2)
            plt.show()
            
    tX = tX_logs
    return tX

def normalize_data(tX):
    """Perform normalization of data"""
    
    #temp_mean = []
    #temp_min_max = []
    
    for i in range(tX.shape[1]):
        #temp_mean.append(np.mean(tX[:,i]))
        #temp_min_max.append((np.max(tX[:,i]) - np.min(tX[:,i])))
        tX[:,i] = tX[:,i] - np.mean(tX[:,i])
        tX[:,i] = tX[:,i] / (np.max(tX[:,i]) - np.min(tX[:,i]))
    
    #tX[:,22] = temp_min_max[22]*tX[:,22]
    #tX[:,22] += temp_mean[22]
    
    return tX

def personalized_transform(tX, correlation=False):
    """Apply transformation, depending on the 
       type of distribution"""
    
    if correlation == False:
        
        # transform the features with logarithm
        list_1 = [0,2,3,8,9,10,11,12,13,16,19,21,23,26,29]
        tX[:, list_1] = log_transform(tX[:, list_1])
        
        # normalize data
        list_2 = [4,5,6,7,14,24,25,27,28]
        tX[:, list_2] = normalize_data(tX[:, list_2])
        
    return tX
    
    if correlation == True:
        
        # Remove correlated features
        tX_no_correlation = np.delete(tX, [21,29], axis=1)
        features_names = np.delete(features_names, [21,29])
        
        # transform the features with logarithm
        list_1 = [0,2,3,8,9,10,11,12,13,16,19,22,25,28] #Change the lists
        tX[:, list_1] = log_transform(tX[:, list_1]) 
        
        # normalize data
        list_2 = [4,5,6,7,14,23,24,26,27] #Change the lists
        tX[:, list_2] = normalize_data(tX[:, list_2])
        
    return tX 


def check_correlation(tX):
    """Study correlation between data
       Only data with more than 0.9 
       correlation score are plotted"""
    
    a = []
    temp = 0
    for i in range(tX.shape[1]):
        for j in range(tX.shape[1]):
            print(features_names[i])
            if i != j:
                temp = np.corrcoef(tX[:,i],tX[:,j])
                a.append(temp)
                if abs(temp[0,1]) > 0.9:
                    print("Correlation scores for feature {}: ".format(features_names[i+2]))
                    print(features_names[i+2],features_names[j+2])
                    print(temp)
                    u.append(temp)
                    plt.scatter(tX[:,i],tX[:,j])
                    plt.show()
        a = []