from models.Model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import validation.CrossValidate as CV


# This model contains the code for a RandomForest model for the Titanic task, including
# feature selection, training and testing methods.
class RF(Model):
    def __init__(self, params):
        self.params = params
        self.train_set_size = -1
        # used for the name of the prediction file
        self.name = "RF"

    def feature_selection(self, x_train, y_train):
        # get all names of the features
        # self.featureList = list(x_train.columns.values)

        # we only want numerical variables
        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)

        # TODO: remove this later
        # Check feature importances for basic classifier
        # import auxiliary.feature_importance as fs
        # clf = RandomForestClassifier()
        # clf.fit(x_train[self.featureList], y_train)
        # fs.analyze_feature_importance(clf, self.featureList)

        # print( "Feature list after feature selection:" )
        # print(self.featureList)

        self.featureList = ['Pclass', 'Sex', 'Age', 'Fare_PP', 'Embarked', 'Age*Class',
        'Ticket_firstchar', 'FamilySize', 'Embarked_1', 'Embarked_2', 'Embarked_3', 'Title_1', 'Title_2'
        , 'Title_3', 'Title_4', 'Title_5']
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
        clf_raw = RandomForestClassifier()
        # param_grid = {'max_features': [1, int(np.sqrt(len(self.featureList))), len(self.featureList)],
        #               'max_depth': [3, None],
        #               'min_samples_split' :[2, 3, 10],
        #               'min_samples_leaf' : [1, 3, 10],
        #               'criterion':['gini', 'entropy'],
        #               'bootstrap':[True, False]}

        # best model so far
        param_grid = {'max_features': [4],
                      'max_depth': [None],
                      'min_samples_split' :[10],
                      'min_samples_leaf' : [10],
                      'criterion':['gini'],
                      'bootstrap':[True]}

        # find best parameters
        self.clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10, scoring="accuracy", n_jobs=2)
        self.clf.fit(train_X, train_Y)

        print("Best parameters:")
        print(self.clf.best_params_)

        # print best performance of best model of gridsearch with cv
        self.acc = self.clf.best_score_
        print("Model with best parameters, train set avg CV accuracy:", self.acc)


        # # Cross-Validation to get performance estimate
        # # NOTE: this gives a performance indication of clf_raw, not clf with the optimal parameters from the gridsearch
        # clf_raw = RandomForestClassifier(bootstrap=True, min_samples_leaf=3, min_samples_split=10, criterion='gini', max_features=4,max_depth=None, n_estimators=250)
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
