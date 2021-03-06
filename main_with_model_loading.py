import argparse, sys
import pandas as pd
from os.path import isfile
import auxiliary.modelPlots as plottery
import auxiliary.outlierDetection as outliers
from input_output.TitanicLoader import TitanicLoader
from preprocessing.TitanicPreprocessor import TitanicPreprocessor
from featureEngineering.TitanicFeatures import TitanicFeatures
from input_output.TitanicSaver import TitanicSaver
from validation.testPerformance import testAccuracy
from ensembles.votingEnsemble import VotingEnsemble
from models.RandomForestModel import RF
from models.SVMModel import SVM
from models.KNNModel import KNN
from models.MLPModel import MLP
from models.GRBTModel import GRBT
from models.GPModel import GP
from models.XGBoostModel import XGBoost
from models.BayesModel import Bayes
from models.AdaBoostModel import AdaBoostModel as AdaBoost
from models.ExtraTreesModel import ExtraTreesModel as ET
from models.LogitRegModel import LogitRegModel as Logit


class Pipeline(object):
    def __init__(self, loader=TitanicLoader, preprocessor=TitanicPreprocessor, features=TitanicFeatures,
             models=[RF], saver=TitanicSaver, training_mode=False):
        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()
        self.params = None
        self.model_params = None

        self.training_data_file = "Data/train.csv"
        self.test_data_file = "Data/test.csv"

        self.loader = loader()
        self.preprocessor = preprocessor()
        self.features = features()

        self.models = [m(self.params) for m in models]
        self.saver = saver()

        self.training_mode = False

    def run(self):
        # load data. Test_labels are PassengerIds which we need to save for the submission
        x_train, y_train, x_test, test_labels = self.loader.load_split(training_data_file=self.training_data_file, test_data_file=self.test_data_file)

        # detect outliers
        out = outliers.detect_outliers(x_train, 2, ["Age", "SibSp", "Parch", "Fare"])
        print ("Dropping ", len(out) , " outliers")
        # drop outliers
        x_train = x_train.drop(out, axis=0).reset_index(drop=True)
        y_train = y_train.drop(out, axis=0).reset_index(drop=True)

        # process in whole, so the train and test would have the same features (for one-hot encoding for example)
        combined = pd.concat([x_train,x_test],ignore_index=True)
        preprocessed = self.preprocessor.preprocess_datasets([combined])
        engineered = self.features.engineer_features_multiple_ds(preprocessed)[0]

        # Sanity check
        assert len(engineered) == len(x_train) + len(x_test)

        # split data again
        x_train = engineered[0:len(x_train)]
        x_test = engineered[len(x_train):]

        # train all the models
        for model in self.models:
            print ("\nUsing " , model.name)

            # Load model from 'saved_models' folder if not training and if it's available.
            if not self.training_mode and isfile('saved_models/' + model.name + '.pkl'):
                model = self.loader.load_pkl('saved_models/' + model.name + '.pkl')
            # Else, train model from scratch in training or if no saved model is available during testing.
            else:
                # Warning message if we want to load a model but it does not exist.
                if not self.training_mode:
                    print("Saved model not found..")
                # Check which features are optimal for this model, and train the model with them
                model.feature_selection(x_train, y_train)
                model.train(x_train, y_train, self.model_params)

            # Generate predictions for the test set and write to a csv file
            print ("Predicting test set..")
            model.test(x_test, test_labels)
            self.saver.save_predictions(model.predictions, 'predictions/' + model.name + '.csv')

            print("Accuracy on test set is:", testAccuracy(model.name))

        # Save improved models and scores to disk (in training mode)
        if self.training_mode:
            test_accuracies = [ testAccuracy(model.name) for model in self.models ]
            self.saver.save_models(self.models, test_accuracies)

        # Create ensemble from all the trained models, and test the predictions output
        # NOTE: assumes you trained your model with Gridsearch
        ve = VotingEnsemble(params=[], models=self.models)
        ve.feature_selection(x_train, y_train)
        ve.test(x_test, test_labels)
        self.saver.save_predictions(ve.predictions, 'predictions/' + ve.name + '.csv')
        print("Accuracy on test set is:", testAccuracy(ve.name))

        # show accuracies and correlation of models
        # plottery.compareModelAcc(self.models)
        # plottery.plotModelCorrelation(self.models)


if __name__ == '__main__':

    Pipeline(loader=TitanicLoader, preprocessor=TitanicPreprocessor, features=TitanicFeatures,
            models=[RF, KNN, ET, SVM, Logit, Bayes], saver=TitanicSaver,
            training_mode=True).run()

    # works nice: RF, KNN, ET, SVM, Logit, Bayes
    # works nice: RF, KNN, ET, SVM, AdaBoost, Bayes
    # all models = Bayes, GP, GRBT, RF, AdaBoost, Logit, SVM, XGBoost, MLP, KNN, ET
