from models.Model import Model
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import validation.CrossValidate as CV


# This model contains the code for a ExtraTrees model for the Titanic task, including
# feature selection, training and testing methods.
class ExtraTreesModel(Model):
    def __init__(self, params):
        self.params = params
        self.train_set_size = -1
        # used for the name of the prediction file
        self.name = "ET"

    def feature_selection(self, x_train, y_train):
        # get all names of the features
        # self.featureList = list(x_train.columns.values)

        # we only want numerical variables
        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)

        self.featureList = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked', 'Age_cat', 'Fare_cat', 'Age*Class', 'Ticket_firstchar',
            'FamilySize', 'FamilySize_cat', 'IsAlone', 'Title', 'Title_alt',
            'Embarked_1', 'Embarked_2', 'Embarked_3', 'Title_1', 'Title_2',
            'Title_3', 'Title_4', 'Title_5']

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

        # Hyper-parameter tuning
        clf_raw = ExtraTreesClassifier()

#        param_grid = {'max_features': [1, int(np.sqrt(len(self.featureList))), len(self.featureList)],
#                      'max_depth': [3, None],
#                      'min_samples_split' :[2, 3, 10],
#                      'min_samples_leaf' : [1, 3, 10],
#                      'criterion':['gini', 'entropy'],
#                      'bootstrap':[True, False],
#                      'n_estimators': [100, 150, 200, 250, 300]}

#        Best parameters:                  (found in previous run)
        param_grid = {'max_features': [len(self.featureList)], 'max_depth': [None], 'min_samples_split': [2],
                      'min_samples_leaf': [10], 'criterion': ['gini'], 'bootstrap': [True],
                      'n_estimators': [200]}

        # find best parameters
        self.clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring="accuracy", n_jobs=2, verbose=True)
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
