# -*- coding: utf-8 -*-

import numpy as np


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



def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
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



def sigmoid(t):
    """apply sigmoid function on t."""

    s = 1.0/(1.0 + np.exp(-t))
    
    return s



def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    Z = np.matmul(tx, w)
    prediction = sigmoid(Z)
    loss = (-1)*(y.T @ (np.log(prediction))) + ((1 - y).T @ np.log(1 - prediction))
    
    return np.squeeze(loss)



def calculate_gradient(y, tx, w):
    """compute the gradient of loss for logistic regression."""

    cte = tx.shape[1]
    Z = tx @ w
    prediction = sigmoid(Z)
    gradient = (tx.T @ (prediction - y)) * (1.0/cte)
    
    return gradient



def learning_by_gradient_descent(y, tx, w, gamma):
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

    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    
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