import pandas as pd
import numpy as np

class Loader(object):
    def load_preprocess_split(self, training_data_file, test_data_file):
        train, test = self.load_data(training_data_file, test_data_file)
        test_labels = test['PassengerId']
        train = self.preprocess_data(train)
        test = self.preprocess_data(test)
        X_train, Y_train =  self.split_data(train)
        return X_train, Y_train, test, test_labels

    def load_data(self, training_data_file, test_data_file = None):
        # TODO - do we need to use Pandas?
        train = pd.read_csv(training_data_file)
        test = pd.read_csv(test_data_file)
        return train, test

    def preprocess_data(self, data):
        pass

    def split_data(self, train):
        pass

