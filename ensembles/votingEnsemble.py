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

    def test(self, test_X, labels):
        print("\nUsing ", self.name)

        predictions = np.empty( (len(self.models), len(test_X)) )

        # load only the Survived predictions of each model and append to the list
        for i, model in enumerate(self.models):
            df = pd.read_csv('predictions/' + model.name + '.csv')
            df = df.drop('PassengerId', axis=1)
            df.rename(columns = {'Survived':model.name}, inplace = True)
            predictions[i] = np.array(df).ravel()


        # TODO: do some weighting
        weights = [1 for model in self.models]

        # # Tuning weights - first implementation
        # accs = [testAccuracy(model.name) for model in self.models]
        # accs5 = [testAccuracy(model.name)**5 for model in self.models] # Basically would make accuracies of 50% count as nothing and exponentially increase importance of accruacy
        # accs3 = [testAccuracy(model.name)**3 for model in self.models]

        print("Generating predictions..")

        self.predictions = []
        # For every item in the test set, calculate the ensemble prediction
        for i in range(len(test_X)):
            # get predictions of models for this passenger
            preds = predictions[:,i].astype(int)

            # TODO: use weights

            # Option 1: Use most occuring prediction value
            majorityVote = np.bincount(preds).argmax()
            self.predictions.append([labels[i], majorityVote])
            # print(preds , " - ", majorityVote)

            # Option 2: Calc average and round
            # avg = int(np.around(np.mean(preds)))
            # self.predictions.append([labels[i], avg])
            # print(preds , " - ", avg)

            i += 1
