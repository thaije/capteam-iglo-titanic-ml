import argparse
import pandas as pd
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
from models.XGBoostModel import XGBoost
from models.BayesModel import Bayes


class Pipeline(object):
    def __init__(self, loader=TitanicLoader, preprocessor=TitanicPreprocessor, features=TitanicFeatures,
             models=[RF], saver=TitanicSaver):
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

            # Check which features are optimal for this model, and train the model with them
            model.feature_selection(x_train, y_train)
            model.train(x_train, y_train, self.model_params)

            # Generate predictions for the test set and write to a csv file
            print ("Predicting test set..")
            model.test(x_test, test_labels)
            self.saver.save_predictions(model.predictions, 'predictions/' + model.name + '.csv')

            print("Accuracy on test set is:", testAccuracy(model.name))


        # Create ensemble from all the trained models, and test the predictions output
        # NOTE: assumes you trained your model with Gridsearch
        ve = VotingEnsemble(params=[], models=self.models)
        ve.feature_selection(x_train, y_train)
        ve.train(x_train, y_train)
        ve.test(x_test, test_labels)
        self.saver.save_predictions(ve.predictions, 'predictions/' + ve.name + '.csv')
        print("Accuracy on test set is:", testAccuracy(ve.name))

        # show accuracies and correlation of models
        # plottery.compareModelAcc(self.models)
        # plottery.plotModelCorrelation(self.models)


if __name__ == '__main__':
    Pipeline(loader=TitanicLoader, preprocessor=TitanicPreprocessor, features=TitanicFeatures,
                 models=[RF, MLP, SVM, GRBT, XGBoost, KNN, Bayes], saver=TitanicSaver).run()
