import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# compares accuracies of classifiers optimized with gridsearch
def compareModelAcc(classifiers):
    cv_means = []
    cv_std = []
    model_names = []
    for classifier in classifiers:
        model_names.append(model.name)
        cv_means.append(model.acc)
        # cv_std.append(cv_result.std())
        cv_std.append(0)


    cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":model_names})

    g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
    g.set_xlabel("Mean Accuracy")
    g = g.set_title("Cross validation scores")
