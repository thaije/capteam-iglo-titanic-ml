import numpy as np
from IO import Loader
import pandas as pd

class TitanicLoader(Loader):
    def preprocess_data(self, data):
        pass

    def split_data(self, train):
        # Get the true class labels from the training set
        x_train = train.drop(['Survived'], axis=1)
        y_train = np.array(train['Survived'])
        return pd.DataFrame.as_matrix(x_train), y_train
