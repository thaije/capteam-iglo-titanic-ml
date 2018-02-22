from IO import Saver, TitanicSaver
from IO.LuukLoader import LuukLoader
from models import Model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import preprocessing, svm
from sklearn.neural_network import MLPClassifier

import CrossValidate as CV
import pandas as pd

class LuukModel(Model):
    def load_preprocess(self, training_data_file, test_data_file):
        return LuukLoader().load_preprocess_split(training_data_file, test_data_file)

    def train(self, X_train, Y_train, model_args):
        self.X_train = X_train
        self.Y_train = Y_train
        # Model Selection
        clf = RandomForestClassifier()
        # clf = AdaBoostClassifier()
        # clf = svm.SVC()
        # clf = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(100,2))

        # Cross Validation
        self.clf = CV.KFold(X_train, Y_train, clf, 2)
        # clf = CV.RepeatedKFold(X_train, y_train, clf, 10, 10)
        # clf = CV.LeaveOneOut(X_train, y_train, clf)
        # clf = CV.StratifiedKFold(X_train, y_train, clf, 10)

    def test(self, X_test, labels):
        # Prediction
        X_test = pd.DataFrame.as_matrix(X_test)
        y_pred = self.clf.predict(X_test)

        # Write predictions to csv file
        id_offset = len(self.X_train)
        self.predictions = []
        for i, prediction in enumerate(y_pred):
            self.predictions.append([labels[i], prediction])

    def save_predictions(self):
        TitanicSaver().save_predictions(self.predictions, 'submission.csv')
