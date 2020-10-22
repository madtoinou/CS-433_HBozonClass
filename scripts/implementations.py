from helpers import *
import numpy as np

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    # compute and store products of the normal equations
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    # solve the normal equations
    w = np.linalg.solve(a, b)
    # compute the loss
    loss = compute_loss(y, tx, w)
    
    return w, loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    # initialise the weigths
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        grad, _ = compute_gradient(y, tx, w)
        # update w with gradient descent 
        w = w - gamma * grad
        # compute loss 
        loss = compute_loss(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    # initialise the weigths
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient
            gradient, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w with the stochastic gradient
            w = w - gamma * gradient
            # calculate loss
            loss = compute_loss(y, tx, w)
            
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    # compute and store products of the normal equations
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    #solve the normal equations
    w = np.linalg.solve(a, b)
    # compute the loss
    loss = compute_loss(y, tx, w)
    
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent
    /!\ Be sure your tx has the constant 1 feature /!\ """
    
    # initialise the weigths
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss 
        loss = calculate_loss(y, tx, w)
        # compute gradient
        gradient = calculate_gradient_logistic(y, tx, w)
        # update w with gradient descent 
        w = w - gamma * gradient
    
    return w, loss
    
    
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
    /!\ Be sure your tx has the constant 1 feature /!\ """
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss 
        loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        # compute gradient
        gradient = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        # update w with gradient descent 
        w = w - gamma * gradient
    
    return w, loss

