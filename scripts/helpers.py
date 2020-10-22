# -*- coding: utf-8 -*-
""" ### Additional helper functions for project 1 ### """

import numpy as np

""" -- Loss function helper -- """

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """ Calculate the loss using mse or mae. """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)

    
""" -------- Helpers for gradient descend --------"""

def compute_gradient(y, tx, w):
    """Compute the gradient"""
    error = y - tx.dot(w)
    gradient = -tx.T.dot(error) / len(error)

    return gradient, error

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Initiate parameters that will output losses and weights vectors
    ws = [initial_w]
    losses = []
    w = initial_w
    #Gradient descent until max_iters criterion is reached
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    return ws, losses



""" -------- Helpers for stochastic gradient descent -------- """

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

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient"""
    error = y - tx.dot(w)
    gradient = -tx.T.dot(error) / len(error)

    return gradient, error


""" -------- Helpers for LOG regression -------- """

def sigmoid(t):
    """apply sigmoid function on t"""
    return 1.0 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """TRAINING :
    Compute the cost by negative log likelihood"""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))

    return np.squeeze(- loss)


def calculate_gradient_logistic(y, tx, w):
    """compute the gradient of loss"""
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y)

    return gradient


"""--------- FUNCTIONS TO PRE_PROCESS DATA ---------- """


def split_data(prediction, data):
    """jet_num distribution constrain heavily the model by acting as a categorical data, we decide to try to split
    the model into submodels based on this feature
    /!\ jet_num needs to be the last column"""
    
    inds_0 = np.nonzero(data[:, -1] == 0)[0]
    dat_0 = data[inds_0, :]
    pred_0 = prediction[inds_0]

    inds_1 = np.nonzero(data[:, -1] == 1)[0]
    dat_1 = data[inds_1, :]
    pred_1 = prediction[inds_1]
    
    #jet_num = 2 or 3
    inds_2 = np.nonzero(data[:, -1] > 1)[0]
    dat_2 = data[inds_2, :]
    pred_2 = prediction[inds_2]

    return pred_0, pred_1, pred_2, dat_0, dat_1, dat_2, inds_0, inds_1, inds_2

def build_model_cst(prediction, data, log_corr = True):
    """Regression data in matrix form by adding cst feature (w0)."""
    if log_corr:
        prediction[np.where(prediction == -1)] = 0
    y = prediction
    x = data
    len_ = len(y)
    tx = np.c_[np.ones(len_), x]
    return y, tx

def standardize(x):
    """Standardize along features axis, implemented to ignore jet_num features."""
    #Store ignored column
    temp = x[:,-1]
    x =  np.delete(x, -1, axis=1)
    #Mean feature-wise
    mean_col = np.mean(x, axis=0)
    #Mean = 0
    x = x - mean_col
    #STD feature-wise
    std_col = np.std(x, axis=0)
    #Std = 1
    x[:, std_col > 0] = x[:, std_col > 0] / std_col[std_col > 0]
    x = np.c_[x, temp]
    return x

def clean(data):
    """Feature data cleaner, replace outliers with feature mean, build normal distribution
    -Added column remover to avoid empty vector"""
    data = np.where(data == -999, np.nan, data)
    
    # Delete columns with nan values
    data = data[:, ~np.all(np.isnan(data), axis=0)]
    # Delete columns with same values
    data = data[:, ~np.all(data[1:] == data[:-1], axis=0)]
    
    # Mean vector column-wise
    col_mean = np.nanmean(data, axis=0)

    # Find indices where Nan appears
    inds = np.where(np.isnan(data))

    # Place column means at each found index
    data[inds] = np.take(col_mean, inds[1])
    return data

def process_data(data, degree):
    """ Concatenate every preprocessing steps into one"""
    #TO IMPLEMENT >> log transformation of right-sqd data
    data = clean(data)
    data = build_poly(data, degree)
    return standardize(data)

def process_and_build(dataX, dataY, degree):
    x = process_data(dataX, degree)
    y, tx = build_model_cst(dataY, x)

    return y, tx

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

"""--------- FUNCTIONS TO COMPUTE SPLIT PREDICITONS ---------- """

from proj1_helpers import predict_labels

def predict(weights0, weights1, weights2, tx0_test, tx1_test, tx2_test, y_test, ids_test_0, ids_test_1, ids_test_2):
    """predict and reorder with test set and
    return the predicted vector"""
    y_pred0 = predict_labels(weights0, tx0_test)
    y_pred1 = predict_labels(weights1, tx1_test)
    y_pred2 = predict_labels(weights2, tx2_test)

    y_pred = np.zeros(len(y_test))

    for i, ind in enumerate(ids_test_0):
        y_pred[ind] = y_pred0[i]

    for i, ind in enumerate(ids_test_1):
        y_pred[ind] = y_pred1[i]

    for i, ind in enumerate(ids_test_2):
        y_pred[ind] = y_pred2[i]

    return y_pred
