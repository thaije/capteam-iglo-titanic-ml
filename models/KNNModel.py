from models.Model import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# This model contains the code for a k-nearest neihgbours model for the Titanic task, including
# training and testing methods.
class KNN(Model):
    def __init__(self, params):
        self.params = params
        self.train_set_size = -1
        # used for the name of the prediction file
        self.name = "KNN"

    def feature_selection(self, x_train, y_train):
        # we only want numerical variables
        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)

        # remove non-categorical or duplicate features
        self.featureList = ['Pclass', 'Sex', 'Embarked', 'Age_cat', 'Fare_cat', 'Age*Class', 'Ticket_firstchar', 'FamilySize', 'FamilySize_cat', 'IsAlone', 'Title']
        print (self.featureList)

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
        # Hyper-parameter tuning
        clf_raw = KNeighborsClassifier()
        param_grid = {'n_neighbors': [3,4,5,6,7],
                      'weights': ['distance', 'uniform'],
                      'algorithm': ['kd_tree', 'brute']}

        # find best parameters
        self.clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring="accuracy", n_jobs=2)
        self.clf.fit(train_X, train_Y)

        print("Best parameters:")
        print(self.clf.best_params_)

        # print best performance of best model of gridsearch with cv
        self.acc = self.clf.best_score_
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
        self.predictions = []
        for i, prediction in enumerate(y_pred):
            self.predictions.append([labels[i], prediction])
        pass
