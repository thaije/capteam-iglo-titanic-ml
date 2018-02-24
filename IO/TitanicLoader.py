import numpy as np
from IO.Loader import Loader
import pandas as pd


# Loader specific for the Titanic task
# The loader loads the data
class TitanicLoader(Loader):

    def load_split(self, training_data_file, test_data_file):
        train, test = self.load_data(training_data_file, test_data_file)
        test_labels = test['PassengerId']
        X_train, Y_train =  self.split_data(train)

        print( "\n" + ('-' * 40) )
        print( " Original data")
        print( '-' * 40)
        print( X_train.head() )

        return X_train, Y_train, test, test_labels

    def split_data(self, train):
        # split the features and predector feature
        train_X = train
        train_Y = train_X["Survived"]
        del train_X["Survived"]
        return train_X, train_Y
