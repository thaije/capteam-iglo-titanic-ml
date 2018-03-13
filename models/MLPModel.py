# Base object, which can be used as a model for any task
# The model contains the feature selection for this model, the training and test methods
from models.Model import Model
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

class MLP(Model):
    def __init__(self, params):
        self.params = params
        self.featureList = []
        self.acc = -1
        # NOTE: change this to the name of your model, it is used for the name of the prediction output file
        self.name = "MLP"

    def feature_selection(self, x_train, y_train):
        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)

        return self.featureList
    # Training data should probably be a split of features and labels [X, Y]
    def train(self, train_X, train_Y, model_args):
        if self.featureList == []:
            raise ValueError('No features selected. Please first run feature selection.')

        # save trainingset size, prepare the data, and select the features
        self.train_set_size = len(train_X)
        train_X = np.array(train_X[self.featureList])
        train_Y = np.array(train_Y)

        print("Training model..")
        param_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'solver': ['lbfgs','sgd','adam'],
                      'hidden_layer_sizes': [(5,2),(10,2),(5,3,2)],
                      'alpha': [1e-5],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'],
                      }

        clf_raw = MLPClassifier(random_state=1)
        # clf_raw = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
        self.clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring="accuracy", n_jobs=2)

        self.clf.fit(train_X, train_Y)
        print("Best parameters:")
        print (self.clf.best_params_)
        self.acc = self.clf.best_score_

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
