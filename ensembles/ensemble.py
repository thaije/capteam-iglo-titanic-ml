# Base object, which can be used as a ensemble for any task
# This object contains the feature selection, the training and test methods
class Ensemble(object):
    def __init__(self, params, models):
        self.params = params
        self.featureList = []
        self.acc = -1
        # NOTE: change this to the name of your ensemble, it is used for the name of the prediction output file
        self.name = "baseEnsemble"
        self.models = models

    def feature_selection(self, x_train, y_train):
        pass

    # Training data should probably be a split of features and labels [X, Y]
    def train(self, x, y, model_args):
        pass

    def test(self, X_test, labels):
        pass
