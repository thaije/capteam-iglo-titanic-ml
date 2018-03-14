from models.Model import Model
from sklearn.model_selection import GridSearchCV
import numpy as np
import validation.CrossValidate as CV
from sklearn.naive_bayes import GaussianNB

# This model contains the code for a GradientBoostingClassifier model for the Titanic task, including
# feature selection, training and testing methods.
class Bayes(Model):
    def __init__(self, params):
        self.params = params
        self.train_set_size = -1
        # used for the name of the prediction file
        self.name = "Bayes"

    def feature_selection(self, x_train, y_train):
        # we only want numerical variables
        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)

        self.featureList.remove('Age')
        self.featureList.remove('SibSp')
        self.featureList.remove('Parch')
        self.featureList.remove('Fare')
        self.featureList.remove('Title_alt')
        self.featureList.remove('hasCabinData')
        self.featureList.remove('nameLength')

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
        clf_raw = GaussianNB()
        param_grid = {'priors': [None]
                      }

        from sklearn.model_selection import cross_val_score, StratifiedKFold
        kfold = StratifiedKFold(n_splits=10)
        cv = cross_val_score(clf_raw, train_X, y = train_Y, scoring = "accuracy", cv = kfold, n_jobs=4)
        print ("Mean: ", cv.mean(), " std:", cv.std())

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
