# -*- coding: utf-8 -*-
"""All the implementations"""

import numpy as np
import csv

"""IMPLEMENTATIONS"""

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm.
       Return the updated loss and weights."""
    
    w = initial_w
    
    for n_iter in range(max_iters):
        error, gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma*gradient
    
    return w, loss


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm.
       Return the updated loss and weights."""
    
    w = initial_w

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            error, gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w - gamma*gradient
    
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution using normal equations.
       Returns the loss and weights."""
    
    #version original
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)  
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Calculate the ridge regression solution using normal equations.
       Returns the loss and weights."""
    
    diag = np.eye(tx.shape[1])
    cte = 2*tx.shape[0] 
    A = tx.T @ tx + lambda_ * cte * diag
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)
    
    return w, loss


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the updated loss and weights.
    """

    loss = compute_log_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    
    return w, loss


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the updated loss and weights.
    """

    loss, grad, hess = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma*(np.linalg.inv(hess) @ grad)
    
    return w, loss


"""UTILITARIES"""


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
            


def compute_mse(y, tx, w):
    """Calculate the loss using mse."""
    
    losses = np.empty((y.shape))
    error = y - np.dot(tx,w)
    cte = 1/(2*y.shape[0])
    losses = cte*np.dot(np.transpose(error),error)
    
    return losses



def compute_mae(y, tx, w):
    """Calculate the loss using mae"""

    losses = np.empty((y.shape))
    error = y - np.dot(tx,w)
    cte = 1/(y.shape[0])
    losses = cte*np.absolute(error)
    
    return losses



def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    gradient = np.empty((y.shape))
    error = y - np.dot(tx,w)
    cte = 1/(y.shape[0])
    gradient = -cte*(tx.T @ error)
    
    return error, gradient



def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n          labels."""
    
    gradient = np.empty((y.shape))
    error = y - np.dot(tx,w)
    cte = 1/(y.shape[0])
    gradient = -cte*(tx.T @ error)
    
    return error, gradient



def compute_subgradient(y, tx, w):
    """Compute the subgradient."""
    
    error = y - (tx @ w)
    subgradient = -(tx.T @ np.sign(error)) / error.shape[0]
    
    return error, subgradient



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data        'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y`     and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the       randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
            
def sigmoid(t):
    """apply sigmoid function on t."""

    s = 1.0/(1.0 + np.exp(-t))    
    return s



def compute_log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1-pred))
    return np.squeeze(- loss)



def calculate_gradient(y, tx, w):
    """compute the gradient of loss for logistic regression."""

    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred.reshape(y.shape[0]) - y)
    return grad



def learning_by_gradient_descent_stoch(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """

    loss = compute_log_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma*grad
    
    return w, loss



def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""

    prediction = sigmoid(tx @ w)
    S = np.zeros((tx.shape[0],tx.shape[0]))
    
    for i in range(tx.shape[0]):
        S[i,i] = prediction[i]*(1-prediction[i])
    hess = tx.T @ S @ tx
    
    return hess



def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""

    loss = compute_log_loss(y, tx, w)/tx.shape[0]
    grad = calculate_gradient(y, tx, w)/tx.shape[0]
    hess = calculate_hessian(y, tx, w)/tx.shape[0]
    
    return loss, grad, hess



def learning_by_newton_method(y, tx, w, gamma):   #Add gamma or not?
    """ Do one step on Newton's method.
        return the loss and updated w."""

    loss, grad, hess = logistic_regression(y, tx, w)
    w -= gamma*(np.linalg.inv(hess) @ grad)
    
    return loss, w



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""

    reg_param = (lambda_ / 2)*(w.T @ w)
    loss = compute_log_loss(y, tx, w) + reg_param
    grad = calculate_gradient(y, tx, w) + lambda_* w
    lambdas = np.zeros((w.shape[0],w.shape[0]))
    np.fill_diagonal(lambdas, lambda_)
    hess = calculate_hessian(y, tx, w) + lambdas
    
    return loss, grad, hess       
            

"""PREPROCESSING"""    
    
    
def split_jet_num(y, tx):
    """
    jet_num distribution constrain heavily the model by acting as a categorical data, we       decide to try to split
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



def cols_log_transform(tx, cols_idx=[0,2,3,5,8,9,10,13,16,19,21,22,23]):
    """Apply transformation, depending on the 
       type of distribution"""
    
    #list of identified features that would benefit a log-transform
    #22 correspong to PRE_tau_pt (swapped with jet_num)
    # [0,2,3,5,8,9,10,13,16,19,21,22,23]
    # transform the features with logarithm
    tx[:, cols_idx] = np.log(1+tx[:, cols_idx])

    return tx



def standardize_matrix(tx, mean=[], std=[]):
    """
    Standardize along features axis and stores mean and std.
    """
    #no exsiting params
    if len(mean) == 0 and len(std) ==0:
        #shift mean
        nan_mean = np.nanmean(tx, axis=0)
        tx = tx - nan_mean

        #STD feature-wise
        std_col = np.nanstd(tx, axis=0)

        # shift standard deviation to 1
        tx[:, std_col > 0] = tx[:, std_col > 0] / std_col[std_col > 0]
        return tx, nan_mean, std_col
    #known params
    else:
        #shift using training mean
        tx = tx - mean

        # shift using training standard deviation
        tx[:, std > 0] = tx[:, std > 0] / std[std > 0]
        
        return tx, mean, std

    
    
def drop_nan_col(tx):
    """
    Remove the columns containing only NaN values or 0 from the matrix
    """
    
    nan_cols = np.all(np.isnan(tx) | np.isclose(tx, 10**-8), axis=0)
    tx = tx[:,~nan_cols]
    
    return tx



def nan_to_median(tx):
    '''Set NaN values to the value of the mean for each feature'''
    
    col_median = np.nanmedian(tx, axis=0)
    idx = np.where(np.isnan(tx))
    tx[idx] = np.take(col_median, idx[1])
    
    return tx
            
    

def preprocessing(dat_i, degree, kept_cols=[], mean=[], std=[], cols_idx=[0,2,3,5,8,9,10,13,16,19,21,22,23]):
    """Receives the features to preprocess and perform log-transform
       polynomial augmentation, standardization and remove columns
       of nans and zeros
    """
    
    #preallocate the array
    dat_f = np.zeros([dat_i.shape[0], (len(kept_cols))*degree +1])
    
    #change the -999 to nan to exclude them from std, mean and median calculations
    dat_i[dat_i == -999] = np.nan

    ##log-transform predetermined features
    dat_f = cols_log_transform(dat_i,cols_idx)
        
    #augment the features matrix using a polynomial basis
    dat_f = build_poly(dat_f,degree)

    #standardize the features matrix (except the constant from build_poly)
    dat_f[:,1:], mean, std = standardize_matrix(dat_f[:,1:], mean=mean, std=std)
    
    #change the nan values to the median of the column
    dat_f = nan_to_median(dat_f)
    
    if len(kept_cols) != 0:
        #temporaire -> virer certaines colonnes, à remonter avant le cols_log_transform
        dat_f[:,:len(kept_cols)] = dat_f[:,kept_cols]
        
    #remove column containing only nans and zeros
    dat_f = drop_nan_col(dat_f)

    return dat_f, mean, std


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
        
    return poly



def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    return np.array(k_indices)


        
def preprocessing(y,tX, degree):
    # --- PREPROCESSING FOR HYPERPARAMETER OPTIMIZATION 3 SPLITS ---
    #putting jet_num in the last column
    tX[tX == -999] = np.nan

    #split data based on jet_num
    """WARNING tX must be unprocessed data so that the processing can
    be jet_num specific WARNING""" 
    #remove the jet_num column -> change the shape of the training set
    pred_0, pred_1, pred_2, dat_0, dat_1, dat_2, inds_0, inds_1, inds_2 = split_jet_num(y,tX)

    ##prepocessing
    dat_0 = cols_log_transform(dat_0)
    dat_1 = cols_log_transform(dat_1)
    dat_2 = cols_log_transform(dat_2)

    """we don't have the same shape because of jet_num removal
    instead of (shape-1)*deg + 2 -> shape*deg +1"""
    dat_0_ = np.zeros([dat_0.shape[0], (dat_0.shape[1])*degree[0] +1])
    dat_1_ = np.zeros([dat_1.shape[0], (dat_1.shape[1])*degree[1] +1])
    dat_2_ = np.zeros([dat_2.shape[0], (dat_2.shape[1])*degree[2] +1])

    dat_0_ = build_poly(dat_0,degree[0])
    dat_1_ = build_poly(dat_1,degree[1])
    dat_2_ = build_poly(dat_2,degree[2])

    #we don't standardize the first column because its the constant
    #introduced by the build_poly
    dat_0_[:,1:] = normalize_data_std(dat_0_[:,1:])
    dat_1_[:,1:] = normalize_data_std(dat_1_[:,1:])
    dat_2_[:,1:] = normalize_data_std(dat_2_[:,1:])

    dat_0_ = nan_to_medi(dat_0_)
    dat_1_ = nan_to_medi(dat_1_)
    dat_2_ = nan_to_medi(dat_2_)
    
    #remove column with nan
    dat_0_ = drop_nan_col(dat_0_)
    dat_1_ = drop_nan_col(dat_1_)
    dat_2_ = drop_nan_col(dat_2_)

    return pred_0, pred_1, pred_2, dat_0_, dat_1_, dat_2_, inds_0, inds_1, inds_2