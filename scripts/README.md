# Machine learning (CS-433)

_____________________________________________________________________

#### Project 1: Higgs Boson detection challenge

_Antoine Madrona, Loïc Bruchez, Valentin Bigot_, October 2020

### Dataset

The  dataset is divided into a training and a testing set composed of 250’000 and 568’238 samples respectively and both having 30 features. The training set is paired with labels where each sample is associated to a category (−1 for  background  noise  and 1 for the presence of a Higgs Boson).

### Model's basic operations

The model implemented in `run.py` loads the training set provided in the DATA_TRAIN_PATH (see Code execution) and several preprocessing steps are performed on this dataset in this order: 
1. Splitting of the features based on the PRI_jet_num categories (0,1 or 2&3) 
2. logarithmic transformation of selected features 
3. Polynomial augmentation of the features 
4. Standardization of the features. 
    
Afterwards, the model is trained using the ridge regression algorithm and weights are obtained. These weights are used to predict each labels of the splitted dataset and the predictions are finally merged and the submission file is created.

### Useful files

The code is separated in 3 distinctive files containing all the functions to reproduce our results:

>1. implementations.py
>2. run.py


#### implementations.py:

This file contains all the functions required to reproduce our preprocessing pipeline and to use our regression model. Specifically, the file is separated in 3 sections:

- "IMPLEMENTATIONS" is composed of the 6 functions `least_squares_GD`, `least_squares_SGD`, `least_squares`, `ridge_regression`, `logistic regression` and `reg_logistic_regression` constituing a toolbox for development of the regression model.

- "UTILITARIES" contains complementary functions to ensure good working of the methods present in "IMPLEMENTATION" section, as well as functions needed for prediction and loading of datasets.

- "PREPROCESSING" contains all the preprocessing steps used in this work to optimize the model's performance.


#### run.py:

`run.py` allows to reproduce the best prediction accuracy stated in the report. The optimal hyperparameters are already provided. Function `load_csv_data()` provided by the teachers for loading train set, predict labels and create a submission file in `.csv` format is also given in this file.


### Code execution

1) Downoload and unzip the `.zip` folders `train.csv` and `test.csv` at https://github.com/epfml/ML_course/tree/master/projects/project1/data

2) Set the DATA_TRAIN_PATH and DATA_TEST_PATH with your own path (e.g. '../data/train.csv', '../data/test.csv') in the `run.py` file, all the optimal hyperparameters are already provided

3) Set the OUTPUT_PATH (e.g. '../sub.csv') to define where the submission file must be saved

3) Run the following command line in the terminal : python3 run.py, to obtain the `.csv` file for submission
