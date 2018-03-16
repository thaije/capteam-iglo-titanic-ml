from models.Model import Model
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn import svm

class SVM(Model):

    def __init__(self, params):
        self.params = params
        self.featureList = []
        self.acc = -1
        # used for the name of the prediction file
        self.name = "SVM"

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
            # {'C': [0.01, 0.1, 1], 'kernel': ['linear']}
            {'C': [0.1], 'kernel': ['linear']} # just use best of previous grid search
        ]

        # probability=True makes it slower, is needed for it to work with the ensemble learning
        clf_raw = svm.SVC(probability=True)
        self.clf = GridSearchCV(clf_raw, param_grid, cv=10, scoring="accuracy", n_jobs=2)

        self.clf.fit(train_X, train_Y)
        print (self.clf.best_params_)
        self.acc = self.clf.best_score_

        # Cross Validation
      #  self.clf, self.acc, scores = CV.KFold(train_X, train_Y, clf, 4)
        # self.clf, optimalScore, scores = CV.RepeatedKFold(X_train, y_train, clf, 10, 10)
        # self.clf, optimalScore, scores = CV.LeaveOneOut(X_train, y_train, clf)
        # self.clf, optimalScore, scores = CV.StratifiedKFold(X_train, y_train, clf, 10)

        print("Model with best parameters, average accuracy over K-folds:", self.acc)


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
