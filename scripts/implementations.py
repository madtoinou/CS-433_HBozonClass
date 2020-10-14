# -*- coding: utf-8 -*-
"""functions implementations"""
import numpy as np


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm.
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):

        # update w by gradient
        w = w - gamma*compute_gradient(y, tx, w)

    return compute_loss_mse(y, tx, w), w
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm.
    """
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, 
                                            batch_size=1, 
                                            num_batches=1
                                           ):
            # compute gradient
            gradient = compute_gradient(y_batch, tx_batch, w)
            # update w by gradient
            w = w - gamma*gradient

    return compute_loss_mse(y, tx, w), w
    
def least_squares(y,tx):
    """calculate the least squares solution.
    """
    w_star = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    return (y-tx.dot(w_star)).sum(), w_star

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    """
    w = np.linalg.solve(tx.T@tx + np.eye(len(tx[0]))*lambda_*2*len(tx), tx.T@y)
    return compute_loss_mse(y, tx, w), w
    
#def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
#def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    
def compute_loss_mse(y, tx, w):
    """Calculate the loss using mse.
    """
    e = y - tx@w.T
    return (1/(2*len(tx)))*(e*e).sum()

def compute_loss_mae(y, tx, w):
    """Calculate the loss using mae.
    """
    e = np.abs(y - tx@w.T)
    return e.sum()/len(tx)

def compute_gradient(y, tx, w):
    """Compute the gradient.
    """
    e = y - tx@w.T
    return (-1/len(tx))*tx.T@e

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their
    corresponding y_n labels.
    """
    e = y - tx@w.T
    return (-1/len(tx))*tx.T@e

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
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

# ***

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i, val_0 in enumerate(w0):
        for j, val_1 in enumerate(w1):
            losses[i][j] = compute_loss(y, tx, np.array([val_0, val_1]))
    return losses

def get_best_parameters(w0s, w1s, losses):    
    min_loss = np.amin(losses)
    idx_w0, idx_w1 = np.where(losses == min_loss)
    return min_loss, w0x[idx_w0], w1[idx_w1]


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    p = np.random.permutation(len(x))
    limit= int(np.ceil(len(x)*ratio))
    # x_tr, y_tr, x_te, y_te
    return x[p[:limit]], y[p[:limit]], x[p[limit:]], y[p[limit:]]

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train
    te_idx = k_indices[k]
    tr_idx = list(set(range(len(x))) - set(k_indices[k]))
        
    # form data with polynomial degree
    x_tr = build_poly(x[tr_idx], degree)
    x_te = build_poly(x[te_idx], degree)
    
    # ridge regression
    _, w_k = ridge_regression(y[tr_idx],x_tr,lambda_)
        
    loss_tr = np.sqrt(2* compute_loss_mse(y[tr_idx],x_tr,w_k))
    loss_te = np.sqrt(2* compute_loss_mse(y[te_idx],x_te,w_k))

    return loss_tr, loss_te

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly