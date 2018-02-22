from IO.Loader import Loader


class Model(object):
    def __init__(self, params):
        self.params = params
    def load_preprocess(self, training_data_file, test_data_file):
        return Loader().load_preprocess_split(training_data_file, test_data_file)

    # Training data should probably be a split of features and labels [X, Y]
    def train(self, x, y, model_args):
        pass

    def test(self, X_test, labels):
        pass