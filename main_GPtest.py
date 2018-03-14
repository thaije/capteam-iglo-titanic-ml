import argparse
import auxiliary.modelPlots as plottery
from input_output.TitanicLoader import TitanicLoader
from preprocessing.TitanicPreprocessor import TitanicPreprocessor
from featureEngineering.TitanicFeatures import TitanicFeatures
from models.RandomForestModel import RandomForestModel
from models.SVMModel import SVMModel
from models.KNNModel import KNNModel
from ensembles.votingEnsemble import VotingEnsemble
from input_output.TitanicSaver import TitanicSaver
from models.MLPModel import MLP
from models.GBRTModel import GBRT
from models.Bayes import Bayes
from models.GPfModel import GPf
import copy

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
        # self.models = [RandomForestModel(self.params), SVMModel(self.params)]
        # self.models = [RandomForestModel(self.params),GBRT(self.params), SVMModel(self.params), KNNModel(self.params), MLP(self.params), Bayes(self.params)]
        self.models = [GPf(self.params)]
        self.saver = TitanicSaver()

    def run(self):
        # load data. Test_labels are PassengerIds which we need to save for the submission
        x_train, y_train, x_test, test_labels = self.loader.load_split(training_data_file=self.training_data_file, test_data_file=self.test_data_file)

        # For GP
        x_train_G = copy.deepcopy(x_train)
        x_test_G = copy.deepcopy(x_test)

        # # preprocess the data and do feature engineering. We just add all features
        [x_train, x_test] = self.preprocessor.preprocess_datasets( [x_train, x_test] )
        [x_train, x_test] = self.features.engineer_features_multiple_ds( [x_train, x_test] )


        # train all the models
        for model in self.models:
            print("\nUsing " , model.name)

            # Check which features are optimal for this model, and train the model with them
            if model.name == 'GPf':
                model.train(x_train_G, y_train, self.model_params)

                # Generate predictions for the test set and write to a csv file
                print ("Predicting test set..")
                model.test(x_test_G, test_labels)

            else:
                model.feature_selection(x_train, y_train)
                model.train(x_train, y_train, self.model_params)

                # Generate predictions for the test set and write to a csv file
                print ("Predicting test set..")
                model.test(x_test, test_labels)

            self.saver.save_predictions(model.predictions, 'predictions/' + model.name + '.csv')


        # Create ensemble from all the trained models, and test the predictions output
        # NOTE: assumes you trained your model with Gridsearch
        ve = VotingEnsemble(params=[], models=self.models)
        ve.feature_selection(x_train, y_train)
        ve.train(x_train, y_train)
        ve.test(x_test, test_labels)
        self.saver.save_predictions(ve.predictions, 'predictions/' + ve.name + '.csv')

        # show accuracies and correlation of models
        plottery.compareModelAcc(self.models)
        plottery.plotModelCorrelation(self.models)


if __name__ == '__main__':
    Pipeline().run()
