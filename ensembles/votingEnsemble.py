import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from validation.testPerformance import testAccuracy
from sklearn.model_selection import GridSearchCV
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

        print("\n")

        predictions = []

        # load only the Survived predictions of each model and append to the list
        for model in self.models:
            df = pd.read_csv('predictions/' + model.name + '.csv')
            df = df.drop('PassengerId', axis=1)
            df.rename(columns = {'Survived':model.name}, inplace = True)
            predictions.append(df)

        print (len(predictions))
        print(len(predictions[0]))




        # # Tuning weights - first implementation
        # accs = [testAccuracy(model.name) for model in self.models]
        # accs5 = [testAccuracy(model.name)**5 for model in self.models] # Basically would make accuracies of 50% count as nothing and exponentially increase importance of accruacy
        # accs3 = [testAccuracy(model.name)**3 for model in self.models]
        # param_grid = {'weights': [accs, None, accs5, accs3]}






    def test(self, test_X, labels):
        print("Generating predictions..")

        test_X = np.array(test_X[self.featureList])
        pred_Y = self.clf.predict(test_X)

        # Write predictions to csv file
        self.predictions = []
        for i, prediction in enumerate(pred_Y):
            self.predictions.append([labels[i], prediction])
