from models.Model import Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import validation.CrossValidate as CV


# This model contains the code for a RandomForest model for the Titanic task, including
# feature selection, training and testing methods.
class AdaBoostModel(Model):
    def __init__(self, params):
        self.params = params
        self.train_set_size = -1
        # used for the name of the prediction file
        self.name = "AdaBoost"

    def feature_selection(self, x_train, y_train):
        # get all names of the features
        # self.featureList = list(x_train.columns.values)

        # we only want numerical variables
        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)

        # TODO: remove this later
        # Check feature importances for basic classifier
        # import feature_selection as fs
        # clf = AdaBoostClassifier()
        # clf.fit(x_train[self.featureList], y_train)
        # fs.analyze_feature_importance(clf, self.featureList)

        # print( "Feature list after feature selection:" )
        # print(self.featureList)

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
        DTC = DecisionTreeClassifier()
        clf_raw = AdaBoostClassifier(base_estimator=DTC)
        
        # Best parameters	              (found in a previous run)
        param_grid = {'n_estimators': [200],
                      'learning_rate': [0.1],
                      'algorithm': ['SAMME'],
                      'base_estimator__max_depth': [3],
                      'base_estimator__criterion': ['entropy'],
                      'base_estimator__splitter': ['random'],
                      'base_estimator__max_features': [4],
                      'base_estimator__min_samples_split': [2],
                      'base_estimator__min_samples_leaf': [10]}
        
#        param_grid = {'n_estimators': [100, 200, 300],
#                      'learning_rate': [0.1, 0.001, 0.0001, 0.00001],
#                      'algorithm': ['SAMME', 'SAMME.R'],
#                      'base_estimator__max_depth': [3, None],
#                      'base_estimator__criterion': ['gini', 'entropy'],
#                      'base_estimator__splitter': ['best', 'random'],
#                     # 'base_estimator__max_features': [1, int(np.sqrt(len(self.featureList))), len(self.featureList)],
#                     # 'base_estimator__min_samples_split': [2, 3, 10],
#                     #'base_estimator__min_samples_leaf': [1, 3, 10]}

        # find best parameters
        self.clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring="accuracy", n_jobs=2, verbose=True)
        self.clf.fit(train_X, train_Y)

        print("Best parameters:")
        print(self.clf.best_params_)

        # print best performance of best model of gridsearch with cv
        self.acc = self.clf.best_score_
        print("Model with best parameters, average accuracy over K-folds:", self.acc)


        # # Cross-Validation to get performance estimate
        # # NOTE: this gives a performance indication of clf_raw, not clf with the optimal parameters from the gridsearch
        # clf_raw = AdaBoostClassifier()
        # cv_scores = CV.KFold(train_X, train_Y, clf_raw)
        # print("Best accuracy:", str(np.max(cv_scores)) , ". Mean:", str(np.mean(cv_scores)), "| Std:", str(np.std(cv_scores)))


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
