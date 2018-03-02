from Models.Model import Model
import numpy as np
import pandas as pd
import Validation.CrossValidate as CV

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier

class SVMModel(Model):

    def __init__(self, params):
        self.params = params
        self.featureList = []
        self.acc = -1

    def feature_selection(self, x_train, y_train):
        # we only want numerical variables

        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)

        return self.featureList

    # train the model with the features determined in feature_selection()
    def train(self, train_X, train_Y, model_args):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')

        # save trainingset size, prepare the data, and select the features
        self.train_set_size = len(train_X)
        train_X = np.array(train_X[self.featureList])
        train_Y = np.array(train_Y)

        print("Training model..")
        param_grid = [
            {'C': [0.01, 0.1,1], 'kernel': ['linear']}
        ]

        clf_raw = svm.SVC()
        self.clf = GridSearchCV(clf_raw, param_grid, cv=4, scoring='accuracy')
        self.clf.fit(train_X, train_Y)
        print self.clf.best_params_
        self.acc = self.clf.best_score_

        means = self.clf.cv_results_['mean_test_score']
        stds = self.clf.cv_results_['std_test_score']

        # Cross Validation
      #  self.clf, self.acc, scores = CV.KFold(train_X, train_Y, clf, 4)
        # self.clf, optimalScore, scores = CV.RepeatedKFold(X_train, y_train, clf, 10, 10)
        # self.clf, optimalScore, scores = CV.LeaveOneOut(X_train, y_train, clf)
        # self.clf, optimalScore, scores = CV.StratifiedKFold(X_train, y_train, clf, 10)

        print("Best accuracy:", self.acc , ". Mean:", str(np.mean(means)), "| Std:", str(np.std(stds)))
        pass


    # predict the test set
    def test(self, X_test, labels):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')
        if self.train_set_size == -1:
            raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")

        X_test = np.array(X_test[self.featureList])
        y_pred = self.clf.predict(X_test)

        # Write predictions to csv file
        id_offset = self.train_set_size
        self.predictions = []
        for i, prediction in enumerate(y_pred):
            self.predictions.append([labels[i], prediction])
        pass
