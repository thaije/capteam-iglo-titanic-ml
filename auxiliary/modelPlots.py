import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# compares accuracies of classifiers optimized with gridsearch
def compareModelAcc(models):
    cv_means = []
    cv_std = []
    model_names = []
    for model in models:
        model_names.append(model.name)
        cv_means.append(model.acc)
        # cv_std.append(cv_result.std())
        cv_std.append(0)

    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":model_names})

    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
    plt.show()


def plotModelCorrelation(models):
    predictions = []

    # load only the Survived predictions of each model and append to the list
    for model in models:
        df = pd.read_csv('predictions/' + model.name + '.csv')
        df = df.drop('PassengerId', axis=1)
        df.rename(columns = {'Survived':model.name}, inplace = True)
        predictions.append(df)

    # concatenate and plot predictions
    ensemble_results = pd.concat(predictions, axis=1)
    g = sns.heatmap(ensemble_results.corr(), annot=True)
    g = g.set_title("Correlation of generated predictions of models")
    plt.show()
