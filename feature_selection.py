import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest

# analyze the feature importance in a random forest model
# see: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
def analyze_feature_importance(forest,feature_labels):
    # obtain relative feature importances
    importances = forest.feature_importances_
    # compute standard deviation tree-wise
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    # get the feature indices
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(feature_labels)):
        print("%d. feature %s (%f)" % (f + 1, feature_labels[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(feature_labels)), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(feature_labels)), indices)
    plt.xlim([-1, len(feature_labels)])
    plt.show()

def obtain_feature_scores(scoring_metric,x_data,y_data,feature_labels):
    rating = scoring_metric(x_data,y_data)
    if isinstance(rating,tuple):
        rating, _ = rating
    indices = np.argsort(rating)
    # invert
    indices = indices[::-1]
    for f in range(len(feature_labels)):
        print("%d. feature %s (%f)" % (f + 1, feature_labels[indices[f]], rating[indices[f]]))

    return rating