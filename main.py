import argparse

from input_output.TitanicLoader import TitanicLoader
from preprocessing.TitanicPreprocessor import TitanicPreprocessor
from featureEngineering.TitanicFeatures import TitanicFeatures
from models.RandomForestModel import RandomForestModel
from models.SVMModel import SVMModel
from input_output.TitanicSaver import TitanicSaver

class Pipeline(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()
        self.params = None
        self.model_params = None

        self.training_data_file = "Data/train.csv"
        self.test_data_file = "Data/test.csv"

        self.loader = TitanicLoader()
        self.preprocessor = TitanicPreprocessor()
        self.features = TitanicFeatures()
        self.models = [RandomForestModel(self.params)]
        self.saver = TitanicSaver()

    def run(self):
        # load data. Test_labels are PassengerIds which we need to save for the submission
        x_train, y_train, x_test, test_labels = self.loader.load_split(training_data_file=self.training_data_file, test_data_file=self.test_data_file)

        # preprocess the data and do feature engineering. We just add all features
        [x_train, x_test] = self.preprocessor.preprocess_datasets( [x_train, x_test] )
        [x_train, x_test] = self.features.engineer_features_multiple_ds( [x_train, x_test] )


        # train all the models
        for model in self.models:

            # Check which features are optimal for this model, and train the model with them
            model.feature_selection(x_train, y_train)
            model.train(x_train, y_train, self.model_params)

            # TODO: save cv validation performance or something
            # model.acc = ..


        # TODO: do ensemble learning with the models / select best model?
        bestModel = self.models[0]

        # Generate predictions with the best model / ensemble model
        bestModel.test(x_test, test_labels)
        self.saver.save_predictions(bestModel.predictions, 'submission.csv')



if __name__ == '__main__':
    Pipeline().run()
