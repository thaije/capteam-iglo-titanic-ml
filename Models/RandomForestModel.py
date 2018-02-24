from IO.Loader import Loader
from Models.Model import Model
import numpy as np
import pandas as pd
import Validation.CrossValidate as CV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.neural_network import MLPClassifier

# TODO: test other methods
# clf = AdaBoostClassifier()
# clf = svm.SVC()
# clf = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(100,2))


# This model contains the code for a RandomForest model for the Titanic task, including
# feature selection, training and testing methods.
class RandomForestModel(Model):
    def __init__(self, params):
        self.params = params
        self.train_set_size = -1

    def feature_selection(self, x_train, y_train):
        # get all names of the features
        # self.featureList = list(x_train.columns.values)

        # we only want numerical variables
        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)

        # TODO: do actual feature selection with sklearn or something
        self.featureList.remove('Age*Class')
        self.featureList.remove('SibSp')
        self.featureList.remove('Parch')
        self.featureList.remove('PassengerId')
        self.featureList.remove('Title_alt')
        self.featureList.remove('IsAlone')
        self.featureList.remove('Age')
        self.featureList.remove('Fare')
        self.featureList.remove('FamilySize')


        print( "Feature list after feature selection:" )
        print(self.featureList)

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
        clf = RandomForestClassifier(n_estimators=250)

        # Cross Validation
        self.clf, self.acc, scores = CV.KFold(train_X, train_Y, clf, 4)
        # self.clf, optimalScore, scores = CV.RepeatedKFold(X_train, y_train, clf, 10, 10)
        # self.clf, optimalScore, scores = CV.LeaveOneOut(X_train, y_train, clf)
        # self.clf, optimalScore, scores = CV.StratifiedKFold(X_train, y_train, clf, 10)

        print("Best accuracy:", self.acc , ". Mean:", str(np.mean(scores)), "| Std:", str(np.std(scores)))
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
