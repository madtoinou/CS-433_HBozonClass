import numpy as np
from proj1_helpers import *
from implementations import ridge_regression

# hyperparameters
lambdas =  [2.04*10**-7, 4.89*10**-11, 3.86*10**-9]

degrees = [6, 5, 4]

#no feature selection
feats_to_keeps = [[],[],[]]

OUTPUT_PATH = './submission.csv'

#dataset loading
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

#preallocating the prediction array
pred = np.zeros(tX_test.shape[0])

""" * preprocessing * """

#move jet_num to the last column
tX.T[[22,-1]] = tX.T[[-1,22]]
tX_test.T[[22,-1]] = tX_test.T[[-1,22]]

#splitting according to jet_num
y_0, y_1, y_2, dat_tr0, dat_tr1, dat_tr2, idx_tr0, idx_tr1, idx_tr2 = split_jet_num(y   ,tX)
_  , _  , _  , dat_te0, dat_te1, dat_te2, idx_te0, idx_te1, idx_te2 = split_jet_num(pred,tX_test)

#log-transform, augment features using polynomial basis and standardize the subsets individualy
tx_tr0, tr0_mean, tr0_std = preprocessing(dat_tr0, degrees[0], feats_to_keeps[0])
tx_tr1, tr1_mean, tr1_std = preprocessing(dat_tr1, degrees[1], feats_to_keeps[1])
tx_tr2, tr2_mean, tr2_std = preprocessing(dat_tr2, degrees[2], feats_to_keeps[2])
#using the params identified in the training set
tx_te0, _, _ = preprocessing(dat_te0, degrees[0], feats_to_keeps[0],mean=tr0_mean, std=tr0_std)
tx_te1, _, _ = preprocessing(dat_te1, degrees[1], feats_to_keeps[1],mean=tr1_mean, std=tr1_std)
tx_te2, _, _ = preprocessing(dat_te2, degrees[2], feats_to_keeps[2],mean=tr2_mean, std=tr2_std)

""" * training * """

#ridge regression on the training sets to obtain the weights
weights = [
           ridge_regression(y_0, tx_tr0, lambdas[0])[0],
           ridge_regression(y_1, tx_tr1, lambdas[1])[0],
           ridge_regression(y_2, tx_tr2, lambdas[2])[0]
          ]

""" * prediction * """

#predict the label for each subset independantly
pred_0 = predict_labels(weights[0], tx_te0)
pred_1 = predict_labels(weights[1], tx_te1)
pred_2 = predict_labels(weights[2], tx_te2)

#reorder and merge the predictions based on the index
pred[idx_te0] = pred_0
pred[idx_te1] = pred_1
pred[idx_te2] = pred_2

create_csv_submission(ids_test, pred, OUTPUT_PATH)