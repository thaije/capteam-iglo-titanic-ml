import argparse

from IO.Loader import Loader
from IO.LuukLoader import LuukLoader
from models.LuukModel import LuukModel
from models.Model import Model

class Pipeline(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()
        # TODO - read Model/Loader from args and import dynamically
        self.model_template = LuukModel
        self.training_data_file = "Data/train.csv"
        self.test_data_file = "Data/test.csv"
        self.params = None
        self.model_params = None

    def run(self):
        # loader = LuukLoader()
        # train = loader.preprocess_data(train)
        # test = loader.preprocess_data(train)
        # # If data is already in the right "shape", just pass through
        # # Training/test data should always exist, validation is optional
        # x_train, y_train = loader.split_data(train)

        model = self.model_template(self.params)
        x_train, y_train, x_test, test_labels = model.load_preprocess(training_data_file=self.training_data_file, test_data_file=self.test_data_file)
        model.train(x_train, y_train, self.model_params)
        model.test(x_test, test_labels)
        model.save_predictions()


if __name__ == '__main__':
    Pipeline().run()