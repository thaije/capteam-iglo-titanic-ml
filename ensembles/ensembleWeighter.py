import numpy as np
from validation.testPerformance import testAccuracy
import pandas as pd

def ensembleWeighter(models, preds, mode='majority', pow=1, dropout=False, p_dropout=0.3):
    """
    :param models: models to include in the ensemble (self.models)
    :param preds: predictions for one particular individual for all models
    :param mode: type of ensemble (majority vote, mean, max, prod, soft: weighted by accuracies)
    :param pow: (if mode is soft) weights the accuracies relative to each other: higher power creates more weight to higher accuracy
    :param dropout: (if mode is soft) drops out random models from the ensemble - per prediction [because we like some synchronicity]
    :param p_dropout: (if mode is soft) probability of model dropout
    :return: prediction for one particular individual: [0, 1]
    """
    if mode == 'majority':
        prediction = np.bincount(preds).argmax()
    if mode == 'mean':
        prediction = int(np.around(np.mean(preds)))
    if mode == 'max':
        prediction = int(np.max(preds))
    if mode =='prod':
        prediction = np.dot(preds, preds)
    if mode == 'soft':
        accuracies = [testAccuracy(model.name) for model in models]
        if dropout:
            drops = np.random.choice(2, size=len(accuracies), p = [p_dropout, 1-p_dropout])
            if np.sum(drops) == 0: drops[np.random.randint(len(drops))] = 1
            accuracies = np.multiply(accuracies, drops)
        normAccs = np.power(accuracies,pow) / np.sum(accuracies)
        prediction = int(np.around(np.dot(normAccs, preds)))

    return prediction