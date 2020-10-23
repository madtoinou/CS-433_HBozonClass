# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *


def compute_mse(y, tx, w):
    """Calculate the loss using mse."""
    
    losses = np.empty((y.shape))
    error = y - np.dot(tx,w)
    cte = 1/(2*y.shape[0])
    losses = cte*np.dot(np.transpose(error),error)
    
    return losses



def compute_loss_MAE(y, tx, w):
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
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    
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



def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        error, gradient = compute_gradient(y, tx, w) # Problem if we use (2,1), use (2,)!!!!!!!!!!
        loss = compute_loss(y, tx, w)
        w = w - gamma*gradient
        ws.append(w)
        losses.append(loss)
    #print("min_loss={l}, w_optimal={w}".format(l=loss, w=w))
    
    return losses[-1], ws[-1]



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
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


            
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    
    ws = [initial_w]
    losses = []
    w = initial_w

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for n_iter in range(max_iters):
            error, gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w - gamma*gradient
            ws.append(w)
            losses.append(loss)
    #print("min_loss={l}, w_optimal={w}".format(l=loss, w=w))
    
    return losses[-1], ws[-1]



def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly



def least_squares(y, tx):
    """calculate the least squares solution using normal equations."""
    
    weights_opt = (np.linalg.solve(tx.T @ tx, tx.T @ y))     # When using solve(), numerical inaccuracies are minimized
    cte = (1/(2*len(y)))
    mse = compute_loss(y, tx, weights_opt) 
    
    return weights_opt, loss



def ridge_regression(y, tx, lambda_):
    """implement ridge regression using normal equations."""
    
    diag = np.eye(tx.shape[1])
    cte = 2*tx.shape[0] 
    A = tx.T @ tx + lambda_ * cte * diag
    b = tx.T @ y
    weights_opt = np.linalg.solve(A, b)     # When using solve(), numerical inaccuracies are minimized
    mse = compute_mse(y, tx, weights_opt)
    
    return mse, weights_opt


def split_data(x, y, ratio, seed):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    sp = int(ratio * x.shape[0])
    train_ind = np.random.permutation(np.arange(x.shape[0]))[:sp]
    test_ind = np.random.permutation(np.arange(x.shape[0]))[sp:]

    x_train = x[train_ind]
    x_test = x[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]
    return x_train, x_test, y_train, y_test



def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    
    return np.array(k_indices)


"""
def cross_validation(y, x, k_indices, k, lambda_, degree):
    #return the loss of ridge regression.
    
    x_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
    x_train = x[np.concatenate(([x_train for i,x_train in enumerate(k_indices) if i!=k]), axis=0)]
    y_train = y[np.concatenate(([y_train for i,y_train in enumerate(k_indices) if i!=k]), axis=0)]
    phi_tr = build_poly(x_train, degree)
    phi_te = build_poly(x_test, degree)
    mse, w = ridge_regression(y_train, phi_tr, lambda_)
    loss_tr = np.sqrt(2*compute_mse(y_train,phi_tr,w))
    loss_te = np.sqrt(2*compute_mse(y_test,phi_te,w))
    
    return loss_tr, loss_te, w
"""


def sigmoid(t):
    """apply sigmoid function on t."""

    s = 1.0/(1.0 + np.exp(-t))    
    return s



def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred+10**-5)) + (1 - y).T.dot(np.log(1-pred+10**-5))
    return np.squeeze(- loss)



def calculate_gradient(y, tx, w):
    """compute the gradient of loss for logistic regression."""

    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred.reshape(y.shape[0]) - y)
    return grad


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """

    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    
    return loss, w

def learning_by_gradient_descent_stoch(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """

    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma*grad
    
    return loss, w



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

    loss = calculate_loss(y, tx, w)/tx.shape[0]
    grad = calculate_gradient(y, tx, w)/tx.shape[0]
    hess = calculate_hessian(y, tx, w)/tx.shape[0]
    
    return loss, grad, hess



def learning_by_newton_method(y, tx, w, gamma):   #Add gamma or not?
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """

    loss, grad, hess = logistic_regression(y, tx, w)
    w -= gamma*(np.linalg.inv(hess) @ grad)
    
    return loss, w



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""

    reg_param = (lambda_ / 2)*(w.T @ w)
    loss = calculate_loss(y, tx, w) + reg_param
    grad = calculate_gradient(y, tx, w) + lambda_* w
    #lambdas = np.zeros((w.shape[0],w.shape[0]))
    #np.fill_diagonal(lambdas, lambda_)
    #hess = calculate_hessian(y, tx, w) + lambdas
    
    return loss, grad               # Possible to add hessian calculation but for now it is too costly in memory


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """

    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)    # Don't forget to add hess if needed
    w -= gamma*(np.linalg.inv(hess) @ grad)
    
    return loss, w

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
    
    return np.log(1+tX)

def normalize_data_minmax(tX):
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

def normalize_data_std(tX):
    """Standardize along features axis, implemented to ignore jet_num features."""
    #Mean = 0
    tX = tX - np.nanmean(tX, axis=0)
    #STD feature-wise
    std_col = np.nanstd(tX, axis=0)
    #Std = 1
    tX[:, std_col > 0] = tX[:, std_col > 0] / std_col[std_col > 0]
    return tX

def cols_log_transform(tX):
    """Apply transformation, depending on the 
       type of distribution"""
    
    # transform the features with logarithm
    list_1 = [3,8,9,22,16,19,13]
    tX[:, list_1] = np.log(1+tX[:, list_1])

    return tX

def nan_to_mean(x):
    '''Set NaN values to the value of the mean for each feature'''
    
    x[x == -999] = np.nan
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    
    #if columns of nan, replace them by 0
    #x[np.isnan(x)] =0
    return x

def nan_to_zero(x):
    '''Set NaN values to the value of the mean for each feature'''
    
    x[x == -999] = np.nan
    x[np.isnan(x)] = 0    
    return x

def nan_to_medi(x):
    '''Set NaN values to the value of the mean for each feature'''
    
    x[x == -999] = np.nan
    col_median = np.nanmedian(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_median, inds[1])
    
    #if columns of nan, replace them by 0
    #x[np.isnan(x)] =0
    return x

def drop_nan_col(x):
    col_median = np.nanmean(x, axis=0)
    idx_col_nan = np.where(np.isnan(col_median))[0]
    #non nan cols
    idx_col = list(set([col_i for col_i in range(x.shape[1])])-set(idx_col_nan))
    return x[:,idx_col]

def split_jet_num(prediction, data):
    """jet_num distribution constrain heavily the model by acting as a categorical data, we decide to try to split
    the model into submodels based on this feature
    /!\ jet_num needs to be the last column"""
    
    inds_0 = np.nonzero(data[:, -1] == 0)[0]
    dat_0 = data[inds_0, :-1]
    pred_0 = prediction[inds_0]

    inds_1 = np.nonzero(data[:, -1] == 1)[0]
    dat_1 = data[inds_1, :-1]
    pred_1 = prediction[inds_1]
    
    #jet_num = 2 or 3
    inds_2 = np.nonzero(data[:, -1] > 1)[0]
    dat_2 = data[inds_2, :-1]
    pred_2 = prediction[inds_2]

    return pred_0, pred_1, pred_2, dat_0, dat_1, dat_2, inds_0, inds_1, inds_2

def split_jet_num4(prediction, data):
    """jet_num distribution constrain heavily the model by acting as a categorical data, we decide to try to split
    the model into submodels based on this feature
    /!\ jet_num needs to be the last column"""
    
    inds_0 = np.nonzero(data[:, -1] == 0)[0]
    dat_0 = data[inds_0, :-1]
    pred_0 = prediction[inds_0]

    inds_1 = np.nonzero(data[:, -1] == 1)[0]
    dat_1 = data[inds_1, :-1]
    pred_1 = prediction[inds_1]
    
    #jet_num = 2 or 3
    inds_2 = np.nonzero(data[:, -1] == 2)[0]
    dat_2 = data[inds_2, :-1]
    pred_2 = prediction[inds_2]
    
    inds_3 = np.nonzero(data[:, -1] == 3)[0]
    dat_3 = data[inds_3, :-1]
    pred_3 = prediction[inds_3]

    return pred_0, pred_1, pred_2, pred_3, dat_0, dat_1, dat_2, dat_3, inds_0, inds_1, inds_2, inds_3

#combine the predictions after the jet_num splitting
def pred_jet_num(weights, degree, tx_test):
    
    #putting jet_num in the last column
    tx_test.T[[22,-1]] = tx_test.T[[-1,22]]
    
    #put outlier to nan to facilitate the tratment
    tx_test[tx_test == -999] = np.nan
    
    #prediction array
    pred = np.zeros(tx_test.shape[0])
    
    #spliting based on jet_num
    _, _, _, dat_0, dat_1, dat_2, inds_0, inds_1, inds_2 = split_jet_num(pred, tx_test)
    
    #preprocessing
    dat_0 = cols_log_transform(dat_0)
    dat_1 = cols_log_transform(dat_1)
    dat_2 = cols_log_transform(dat_2)

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
    
    """
    nan_to_mean(dat_0_)
    nan_to_mean(dat_1_)
    nan_to_mean(dat_2_)
    -> 0.803
    """
    
    nan_to_medi(dat_0_)
    nan_to_medi(dat_1_)
    nan_to_medi(dat_2_)
    """
    -> 0.803
    """
    
    #remove column with nan
    dat_0_ = drop_nan_col(dat_0_)
    dat_1_ = drop_nan_col(dat_1_)
    dat_2_ = drop_nan_col(dat_2_)
    
    y_pred_0 = predict_labels(weights[0], dat_0_)
    y_pred_1 = predict_labels(weights[1], dat_1_)
    y_pred_2 = predict_labels(weights[2], dat_2_)
    
    #replacing the prediction in fornt of the original idx
    pred[inds_0] = y_pred_0
    pred[inds_1] = y_pred_1
    pred[inds_2] = y_pred_2
    return pred

#combine the predictions after the jet_num splitting
def pred_jet_num4(weights, degree, tx_test):
    
    tx_test.T[[22,-1]] = tx_test.T[[-1,22]]
    tx_test[tx_test == -999] = np.nan
    
    #prediction array
    pred = np.zeros(tx_test.shape[0])

    #split data based on jet_num
    #remove the jet_num column -> change the shape of the training set
    pred_0, pred_1, pred_2, pred_3, dat_0, dat_1, dat_2, dat_3, inds_0, inds_1, inds_2, inds_3 = split_jet_num4(pred,tx_test)

    ##prepocessing
    dat_0 = cols_log_transform(dat_0)
    dat_1 = cols_log_transform(dat_1)
    dat_2 = cols_log_transform(dat_2)
    dat_3 = cols_log_transform(dat_3)

    """we don't have the same shape because of jet_num removal
    instead of (shape-1)*deg + 2 -> shape*deg +1"""
    dat_0_ = np.zeros([dat_0.shape[0], (dat_0.shape[1])*degree[0] +1])
    dat_1_ = np.zeros([dat_1.shape[0], (dat_1.shape[1])*degree[1] +1])
    dat_2_ = np.zeros([dat_2.shape[0], (dat_2.shape[1])*degree[2] +1])
    dat_3_ = np.zeros([dat_3.shape[0], (dat_3.shape[1])*degree[3] +1])


    dat_0_ = build_poly(dat_0,degree[0])
    dat_1_ = build_poly(dat_1,degree[1])
    dat_2_ = build_poly(dat_2,degree[2])
    dat_3_ = build_poly(dat_3,degree[3])


    #we don't standardize the first column because its the constant
    #introduced by the build_poly
    dat_0_[:,1:] = normalize_data_std(dat_0_[:,1:])
    dat_1_[:,1:] = normalize_data_std(dat_1_[:,1:])
    dat_2_[:,1:] = normalize_data_std(dat_2_[:,1:])
    dat_3_[:,1:] = normalize_data_std(dat_3_[:,1:])


    dat_0_ = nan_to_medi(dat_0_)
    dat_1_ = nan_to_medi(dat_1_)
    dat_2_ = nan_to_medi(dat_2_)
    dat_3_ = nan_to_medi(dat_3_)


    #remove column with nan
    dat_0_ = drop_nan_col(dat_0_)
    dat_1_ = drop_nan_col(dat_1_)
    dat_2_ = drop_nan_col(dat_2_)
    dat_3_ = drop_nan_col(dat_3_)
    
    y_pred_0 = predict_labels(weights[0], dat_0_)
    y_pred_1 = predict_labels(weights[1], dat_1_)
    y_pred_2 = predict_labels(weights[2], dat_2_)
    y_pred_3 = predict_labels(weights[3], dat_3_)

    
    #replacing the prediction in fornt of the original idx
    pred[inds_0] = y_pred_0
    pred[inds_1] = y_pred_1
    pred[inds_2] = y_pred_2
    pred[inds_3] = y_pred_3
    return pred


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
        
def preproc_3split(y,tX):
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

    deg_0 = 3
    deg_1 = 4
    deg_2 = 6

    """we don't have the same shape because of jet_num removal
    instead of (shape-1)*deg + 2 -> shape*deg +1"""
    dat_0_ = np.zeros([dat_0.shape[0], (dat_0.shape[1])*deg_0 +1])
    dat_1_ = np.zeros([dat_1.shape[0], (dat_1.shape[1])*deg_1 +1])
    dat_2_ = np.zeros([dat_2.shape[0], (dat_2.shape[1])*deg_2 +1])

    dat_0_ = build_poly(dat_0,deg_0)
    dat_1_ = build_poly(dat_1,deg_1)
    dat_2_ = build_poly(dat_2,deg_2)

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
    return dat_0_, dat_1_, dat_2_

def preproc_4split(y,tX):
    # --- PREPROCESSING FOR HYPERPARAMETER OPTIMIZATION 4 SPLITS---
    #putting jet_num in the last column
    tX[tX == -999] = np.nan

    #split data based on jet_num
    """WARNING tX must be unprocessed data so that the processing can
    be jet_num specific WARNING""" 
    #remove the jet_num column -> change the shape of the training set
    pred_0, pred_1, pred_2, pred_3, dat_0, dat_1, dat_2, dat_3, inds_0, inds_1, inds_2, inds_3 = split_jet_num4(y,tX)

    ##prepocessing
    dat_0 = cols_log_transform(dat_0)
    dat_1 = cols_log_transform(dat_1)
    dat_2 = cols_log_transform(dat_2)
    dat_3 = cols_log_transform(dat_3)

    deg_0 = 3
    deg_1 = 4
    deg_2 = 3
    deg_3 = 5


    """we don't have the same shape because of jet_num removal
    instead of (shape-1)*deg + 2 -> shape*deg +1"""
    dat_0_ = np.zeros([dat_0.shape[0], (dat_0.shape[1])*deg_0 +1])
    dat_1_ = np.zeros([dat_1.shape[0], (dat_1.shape[1])*deg_1 +1])
    dat_2_ = np.zeros([dat_2.shape[0], (dat_2.shape[1])*deg_2 +1])
    dat_3_ = np.zeros([dat_3.shape[0], (dat_3.shape[1])*deg_3 +1])


    dat_0_ = build_poly(dat_0,deg_0)
    dat_1_ = build_poly(dat_1,deg_1)
    dat_2_ = build_poly(dat_2,deg_2)
    dat_3_ = build_poly(dat_3,deg_3)


    #we don't standardize the first column because its the constant
    #introduced by the build_poly
    dat_0_[:,1:] = normalize_data_std(dat_0_[:,1:])
    dat_1_[:,1:] = normalize_data_std(dat_1_[:,1:])
    dat_2_[:,1:] = normalize_data_std(dat_2_[:,1:])
    dat_3_[:,1:] = normalize_data_std(dat_3_[:,1:])


    dat_0_ = nan_to_medi(dat_0_)
    dat_1_ = nan_to_medi(dat_1_)
    dat_2_ = nan_to_medi(dat_2_)
    dat_3_ = nan_to_medi(dat_3_)


    #remove column with nan
    dat_0_ = drop_nan_col(dat_0_)
    dat_1_ = drop_nan_col(dat_1_)
    dat_2_ = drop_nan_col(dat_2_)
    dat_3_ = drop_nan_col(dat_3_)
    return dat_0_, dat_1_, dat_2_, dat_3_
