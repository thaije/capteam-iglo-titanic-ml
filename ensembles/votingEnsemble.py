import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
np.warnings.filterwarnings('ignore') # annoying np warning in sklearn version


# Settings
# voting = 'soft' = averaging (best when models are close in performance)
# voting = 'hard' = majority vote (best when large gaps in score)
# weighting = None = every model has same weighting
# weighting = e.g. [2,2,1] = give models with higher accuracy more weight, at the moment manual

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

        try:
            # create a list of tuples with the model name and best model estimator from
            # the gridsearch
            for model in self.models:
                modelTupleList.append((model.name, model.clf.best_estimator_))
        except AttributeError:
            print("Error: Ensemble model expects the best estimator under model.clf.best_estimator_ (default location after gridsearch), but could not locate it for " , model.name)
            sys.exit()

        print ("\nTraining voting classifier..")

        # fit the voting classifier
        self.clf = VotingClassifier(estimators = modelTupleList, voting='soft', weights=None, n_jobs=4)
        scores = cross_val_score(estimator = self.clf, X=train_X, y=train_Y, cv=10, scoring='accuracy', n_jobs=1)
        print("10-fold CV over train set: average", np.mean(scores), " std:", np.std(scores), " highest:", np.max(scores))
        self.clf.fit(train_X, train_Y)


    def test(self, test_X, labels):
        print("Generating predictions..")

        test_X = np.array(test_X[self.featureList])
        pred_Y = self.clf.predict(test_X)

        # Write predictions to csv file
        self.predictions = []
        for i, prediction in enumerate(pred_Y):
            self.predictions.append([labels[i], prediction])
