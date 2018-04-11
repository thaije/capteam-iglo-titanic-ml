import numpy as np
from gplearn import functions, fitness, genetic
from sklearn.base import BaseEstimator, ClassifierMixin
from models.Model import Model

def _logloss(true_label, predicted,w):
    eps=1e-15
    loss= 0
    for i in range(len(true_label)):
        p = np.clip(predicted[i], eps, 1 - eps)
        if true_label[i]==1:
            loss += -np.log(p)
        else:
            loss += -np.log(1 - p)
    return loss

def _protected_exponent(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 100, np.exp(x1), 0.)

_exp = functions.make_function(function=_protected_exponent,
                        name='exp',
                        arity=1)

class GP(Model):
    def __init__(self, params):
        self.params = params
        self.train_set_size = -1
        # used for the name of the prediction file
        self.name = "GP"
        self.function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'sin', 'cos', 'tan', _exp]
        self.LOGLOSS = fitness.make_fitness(_logloss, greater_is_better=False)

    def feature_selection(self, x_train, y_train):
        self.featureList = list(x_train.dtypes[x_train.dtypes != 'object'].index)


        print(self.featureList)
        return self.featureList

    # train the model with the features determined in feature_selection()
    def train(self, train_X, train_Y, model_args):
        self.clf = genetic.SymbolicRegressor(population_size=1000,
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.9, p_subtree_mutation=0.1,
                           p_hoist_mutation=0, p_point_mutation=0,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0, function_set=self.function_set, metric=self.LOGLOSS)

        X = np.array(train_X[self.featureList])
        y = np.array(train_Y)
        de = self.clf.fit(X, y)

        print("s-expression of final program:", str(self.clf._program))

        y_gp = np.around(self.clf.predict(X))
        acc_gp = np.sum(y_gp == y) / len(y)
        print("Accuracy GP on train set:", str(acc_gp))

    # predict the test set
    def test(self, X_test, labels):
        # if self.train_set_size == -1:
        #     raise ValueError("Couldn't determine training set size, did you run feature_selection and train first?")
        X_test = np.array(X_test[self.featureList])
        y_pred = self.clf.predict(X_test)
        y_pred = np.around(y_pred)
        # Write predictions to csv file
        self.predictions = []
        for i, prediction in enumerate(y_pred):
            self.predictions.append([labels[i], prediction])
