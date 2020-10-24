# Machine learning (CS-433)

_____________________________________________________________________

#### Project 1: Higgs Boson detection challenge

_Antoine Madrona, Loïc Bruchez, Valentin Bigot_

### Dataset

The  dataset is divided into a training and a testing set composed of 250’000 and 568’238 samples respectively and both having 30 features. The training set is paired with labels where each sample is associated to a category (−1 for  background  noise  and 1 for the presence of a Higgs Boson).

### Useful files

The code is separated in 3 distinctive files containing all the functions to reproduce our results:

>1. implementations.py
>2. visualization.py
>3. run.ipynb

#### implementations.py:

This file contains all the functions required to reproduce our preprocessing pipeline and to use our regression model. Specifically, the file is separated in 5 sections:

- "IMPLEMENTATIONS" is composed of the 6 functions `least_squares_GD`, `least_squares_SGD`, `least_squares`, `ridge_regression`, `logistic regression` and `reg_logistic_regression` constituing a toolbox for development of the regression model.

- "UTILITARIES" contains functions useful for data preprocessing, polynomial expansion as well as complementary functions to ensure good working of the methods present in "IMPLEMENTATION" section.

#### visualization.py:

This file contains the code needed to reproduce all the figures presented in the report.

#### run.ipynb:

`run.ipynb` allows to reproduce the best prediction accuracy stated in the report. The optimal hyperparameters are already provided. Function `load_csv_data()` provided by the teachers for loading train set, predict labels and create a submission file in `.csv` format is also given in this file.


### Code execution

>1) Open run.ipynb

>2) Provide the path where your train set is present to load the prediction vector, feature matrix and IDs of each line of the feature matrix

>3) Run the whole jupyter notebook, no need to set-up the variables (already done).

The whole execution results in the production of a `.csv` file fullfilling the requirements for submission to the Higgs Boson detection challenge on AIcrowd plateform.

