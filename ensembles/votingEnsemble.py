import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier

# Base object, which can be used as a ensemble for any task
# This object contains the feature selection, the training and test methods
class VotingEnsemble(object):
    def __init__(self, params, models):
        self.params = params

        # NOTE: change this to the name of your ensemble, it is used for the name of the prediction output file
        self.name = "VotingEnsemble"
        self.models = models


    def feature_selection(self, train_X, train_Y):
        # we only want numerical variables
        self.featureList = list(train_X.dtypes[train_X.dtypes != 'object'].index)
        return self.featureList


    def train(self, train_X, train_Y):
        train_X = np.array(train_X[self.featureList])
        train_Y = np.array(train_Y)

        modelTupleList = []

        # create a list of tuples with the model name and best model estimator from
        # the gridsearch
        for model in self.models:
            modelTupleList.append((model.name, model.clf.best_estimator_))

        print ("\nTraining voting classifier..")
        # fit the voting classifier
        self.clf = VotingClassifier(estimators=modelTupleList, voting='soft', n_jobs=4)
        self.clf.fit(train_X, train_Y)


    def test(self, test_X, labels):
        print("Generating predictions..")

        test_X = np.array(test_X[self.featureList])
        pred_Y = self.clf.predict(test_X)

        # Write predictions to csv file
        self.predictions = []
        for i, prediction in enumerate(pred_Y):
            self.predictions.append([labels[i], prediction])
