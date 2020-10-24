import numpy as np
from proj1_helpers import *
from implementations import ridge_regression

# hyperparameters
lambdas = [10**-4, 4.06*10**-4, 2.72*10**-7]
degrees = [3, 4, 6]
feats_to_keeps = [
    [0, 2, 10, 20, 38, 21, 23, 3, 32, 16, 19, 37, 1, 18, 29, 54, 26, 43, 7, 28, 13, 39, 41, 34, 52, 8, 44, 25, 17, 24, 27, 4, 42, 31, 35, 14, 9, 45, 50, 22, 51, 53, 47, 40, 48, 12, 15, 6, 33, 11, 30, 36, 5, 46, 49][:49],
    [0, 2, 10, 43, 9, 36, 24, 31, 76, 19, 5, 27, 3, 13, 71, 47, 57, 33, 8, 38, 6, 49, 87, 53, 30, 1, 23, 45, 69, 18, 25, 4, 16, 64, 40, 62, 84, 68, 46, 74, 29, 67, 75, 70, 80, 28, 17, 51, 86, 7, 72, 50, 58, 77, 26, 82, 37, 32, 55, 83, 35, 21, 65, 12, 59, 15, 11, 42, 41, 14, 61, 78, 34, 39, 66, 56, 22, 81, 44, 88, 54, 60, 48, 63, 20, 85, 73, 52, 79][:83],
    [0, 34, 12, 14, 9, 71, 2, 47, 8, 37, 149, 22, 124, 30, 101, 5, 63, 88, 1, 59, 146, 117, 62, 44, 27, 36, 41, 100, 31, 168, 154, 67, 23, 24, 3, 17, 32, 90, 153, 66, 70, 7, 173, 165, 46, 119, 56, 139, 85, 53, 20, 4, 114, 43, 61, 148, 150, 170, 10, 128, 92, 121, 11, 6, 93, 35, 64, 122, 151, 159, 147, 60, 89, 118, 125, 18, 141, 157, 65, 115, 57, 82, 111, 143, 99, 28, 52, 51, 80, 142, 13, 129, 160, 152, 94, 163, 26, 39, 126, 40, 69, 167, 96, 105, 123, 58, 116, 50, 135, 120, 33, 91, 95, 136, 112, 73, 130, 131, 15, 25, 83, 21, 137, 79, 84, 98, 156, 127, 75, 68, 45, 161, 103, 174, 48, 171, 55, 113, 172, 102, 109, 138, 54, 72, 108, 166, 16, 132, 107, 155, 104, 164, 106, 110, 77, 19, 134, 76, 145, 87, 29, 81, 133, 144, 86, 74, 78, 49, 169, 140, 162, 97, 158, 42, 38][:168]
                  ]


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