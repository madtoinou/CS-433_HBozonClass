# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementations import build_poly


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
def split_jet_num(y, tx):
    """
    jet_num distribution constrain heavily the model by acting as a categorical data, we decide to try to split
    the model into submodels based on this feature
    /!\ jet_num needs to be the last column"""
    
    #jet_num == 0
    idx_0 = np.nonzero(tx[:, -1] == 0)[0]
    tx_0  = tx[idx_0, :-1]
    y_0   = y[idx_0]
    
    #jet_num == 1
    idx_1 = np.nonzero(tx[:, -1] == 1)[0]
    tx_1  = tx[idx_1, :-1]
    y_1   = y[idx_1]
    
    #jet_num == 2 or 3
    idx_2 = np.nonzero(tx[:, -1] > 1)[0]
    tx_2  = tx[idx_2, :-1]
    y_2   = y[idx_2]

    return y_0, y_1, y_2, tx_0, tx_1, tx_2, idx_0, idx_1, idx_2

def cols_log_transform(tx):
    """Apply transformation, depending on the 
       type of distribution"""
    
    #list of identified features that would benefit a log-transform
    cols_idx = [3,8,9,13,16,19,22]
    
    # transform the features with logarithm
    tx[:, cols_idx] = np.log(1+tx[:, cols_idx])

    return tx

def standardize_matrix(tx):
    """
    Standardize along features axis.
    """
    #shift mean to 0
    tx = tx - np.nanmean(tx, axis=0)
    
    #STD feature-wise
    std_col = np.nanstd(tx, axis=0)
    
    # shift standard deviation to 1
    tx[:, std_col > 0] = tx[:, std_col > 0] / std_col[std_col > 0]
    return tx

def drop_nan_col(tx):
    """
    Remove the columns containing only NaN values from the matrix
    """
    col_median = np.nanmean(tx, axis=0)
    idx_col_nan = np.where(np.isnan(col_median))[0]
    
    #extract non nan columns
    idx_col = list(set([col_i for col_i in range(tx.shape[1])])-set(idx_col_nan))
    
    return tx[:,idx_col]

def nan_to_median(tx):
    '''Set NaN values to the value of the mean for each feature'''
    
    col_median = np.nanmedian(tx, axis=0)
    idx = np.where(np.isnan(tx))
    tx[idx] = np.take(col_median, idx[1])
    
    return tx
            
def preprocessing(dat_i, degree):
    """
    """
    
    #preallocate the array
    dat_f = np.zeros([dat_i.shape[0], (dat_i.shape[1])*degree +1])
    
    #change the -999 to nan to exclude them from std, mean and median calculations
    dat_i[dat_i == -999] = np.nan

    ##log-transform predetermined features
    dat_tmp = cols_log_transform(dat_i)
    
    #augment the features matrix using a polynomial basis
    dat_tmp = build_poly(dat_tmp,degree)

    #standardize the features matrix (except the constant from build_poly)
    dat_f[:,1:] = standardize_matrix(dat_f[:,1:])

    #remove column containing only nans
    dat_f = drop_nan_col(dat_f)
    
    #change the nan values to the median of the column
    dat_f = nan_to_median(dat_f)

    return dat_f
