import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from matplotlib.backends.backend_pdf import PdfPages

def SS_plot(feature, X):
    """
    :param feature: Categorcal feature to plot
    :param data: dataset
    :return: two plots: percentageplot for survival split and survival probability per category
    """
    fig, ax = plt.subplots(ncols=1, nrows=2)
    # percentageplot(x=feature, data=X, hue="Survived", palette=pal, ax=ax[0])
    sns.countplot(x=feature, data=X, hue="Survived", palette=pal, ax=ax[0])
    plt.tight_layout()
    sns.factorplot(x=feature, y="Survived", data=X, kind="bar", size=6, palette=pal, ax=ax[1])
    ax[1].set_ylabel("Probability(Survive)")
    plt.close()  # Close extra figure it automatically opens
    plt.tight_layout()


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


# Plot palette for all plots
pal = 'Paired'

X = pd.read_csv("train.csv")

# Most basic statistics
stats_table = X.describe # ~38% survived


# Male vs female - obviosu but has to be done
SS_plot(feature="Sex",X=X)

# Title feature
X['Title'] = X.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Dr": "Officer",
    "Rev": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Lady": "Royalty",
    "Mme": "Mrs",
    "Ms": "Mrs",
    "Mrs": "Mrs",
    "Mlle": "Miss",
    "Miss": "Miss",
    "Mr": "Mr",
    "Master": "Master"
}

X['Title'] = X.Title.map(Title_Dictionary)
SS_plot(feature="Title",X=X)

# Ticket feature
X.Ticket = X.Ticket.map(lambda x: x[0]) # This preprocessed Ticket to features of ticket number
SS_plot(feature="Ticket",X=X)

# Age feature
X.loc[X.Age.isnull(), 'Age'] = X.groupby(['Sex','Pclass','Title']).Age.transform('median')

interval = (0, 5, 12, 18, 25, 35, 60, 120)
cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior'] # cats = [0,1,2,3,4,5,6] # numbers for real analysis
X["Age_cat"] = pd.cut(X.Age, interval, labels=cats)

sns.lmplot(x="Age", y= "Survived" , data=X, palette=pal) # Older people are slightly less likely to survive
SS_plot(feature="Age_cat",X=X)

# Fare features
X["Fare"].fillna(X.Fare.mean(), inplace=True)
quant = (0, 8, 15, 31, 600) #intervals to categorize
label_quants = ['quart_1', 'quart_2', 'quart_3', 'quart_4']
X["Fare_cat"] = pd.cut(X.Fare, quant, labels=label_quants)

sns.lmplot(x="Fare", y= "Survived" , data=X, palette=pal) # More fare is more survive
SS_plot(feature="Fare_cat",X=X)

# Plass feature
SS_plot(feature="Pclass",X=X)

# Family features
X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
X['IsAlone'] = 0
X.loc[X['FamilySize'] == 1, 'IsAlone'] = 1

SS_plot(feature="SibSp",X=X)
SS_plot(feature='FamilySize', X=X)

# Embarked feature
SS_plot(feature="Embarked",X=X)

# Overall pairplots for epifany
drops =["PassengerId", "Name", "SibSp", "Parch", "Cabin"]
sns.pairplot(X.drop(drops, axis=1), hue="Survived", size=3, diag_kind="kde")



# Save all images open in one pdf file
multipage('EDA_results.pdf')