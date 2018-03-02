
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# setup 1
# randomForestParams = {'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 3}
# remove PassengerId, SibSp, Parch
# pipeline score = 84.7
# this script score = 83.65

randomForestParams = {'bootstrap': True, 'criterion': 'gini', 'max_depth': None, 'max_features': 4, 'min_samples_leaf': 3, 'min_samples_split': 3}
# or {'bootstrap': True, 'criterion': 'gini', 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 3}
# remove PassengerId, SibSp, Parch
# score = 84.1 / 84.4
# this script 83

# randomForestParams = {'bootstrap': True, 'criterion': 'entropy', 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 3, 'min_samples_split': 10}
# remove PassengerId, SibSp, Parch
# score = 83.8
# this script score: 83.7

# randomForestParams = {'bootstrap': True, 'criterion': 'gini', 'max_depth': None, 'max_features': 4, 'min_samples_leaf': 3, 'min_samples_split': 3}
# remove None
# score = 83.8
# this script score 83.8

# randomForestParams = {'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 1, 'min_samples_leaf': 3, 'min_samples_split': 2}
# remove PassengerId, SibSp, Parch
# score = 84.0
# this script score 83.1

def preprocess(data):
    print ("Preprocessing data..")

    # replace Sex labels with numerical ID
    # ['female','male'] => [0,1]
    data['Sex'].replace(['female','male'], [0,1], inplace=True)

    # Fill in the mean age for values with missing age
    data["Age"].fillna(data.Age.mean(), inplace=True)

    # Fill in missing Fare attributes with the mean
    data["Fare"].fillna(data.Fare.mean(), inplace=True)

    # Fill in missing embarked attributes with most frequent port
    freq_port = data.Embarked.dropna().mode()[0]
    data['Embarked'] = data['Embarked'].fillna(freq_port)

    # replace port numbers with numerical ID
    # ['S','C','Q'] => [1,2,3]
    data["Embarked"].replace(['S','C','Q'], [1,2,3], inplace=True)

    return data


# Bin Age and save it as a new feature
def binAge(data):
    interval = (0, 5, 12, 18, 25, 35, 60, 120)
    cats = [0,1,2,3,4,5,6]
    data["Age_cat"] = pd.cut(data.Age, interval, labels=cats)

    return data


# Put fares into 4 bins (see https://www.kaggle.com/polarhut/titanic-data-science-solutions)
def binFares(data):
    data['Fare_cat'] = data['Fare']
    data.loc[ data['Fare_cat'] <= 7.91, 'Fare_cat'] = 0
    data.loc[(data['Fare_cat'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
    data.loc[(data['Fare_cat'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat']   = 2
    data.loc[ data['Fare_cat'] > 31, 'Fare_cat'] = 3
    data['Fare_cat'] = data['Fare_cat'].astype(int)

    return data


# Create new class Pclass*Age
def combineAgePClass(data):
    data['Age*Class'] = data.Age_cat * data.Pclass
    # data.loc[:, ['Age*Class', 'Age_cat', 'Pclass']].head(10)

    return data

# Create a new feature from the first character of the Ticket
def createTicketFirstChar(data):
    data['Ticket_firstchar'] = data['Ticket']
    data.Ticket_firstchar = data.Ticket_firstchar.map(lambda x: x[0])
    data["Ticket_firstchar"].replace(['A', 'P', 'S', 'C', 'W', 'F', 'L', '1','2','3','4','5','6','7','8','9'], [10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9], inplace=True)

    return data


# Create new feature familysize= self + sibelings/spouse + Parents/children
def createFamilySize(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    return data


# create new feature based on family size, but in categories
def createFamilySizeBinned(data):
    data['FamilySize_cat'] = data['FamilySize']
    data.loc[ data['FamilySize_cat'] <= 1, 'FamilySize_cat'] = 1
    data.loc[(data['FamilySize_cat'] >= 2) & (data['FamilySize_cat'] < 5), 'FamilySize_cat'] = 2
    data.loc[data['FamilySize_cat'] >= 5 , 'FamilySize_cat']   = 3
    # print ( data.groupby('FamilySize_cat')['PassengerId'].nunique() )

    return data


# create new feature which classifies people as alone or not
def createIsAlone(data):
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    return data


# create a new feature which classifies people based on their title,
# with categories "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5
def createTitle(data):
    # first extract the titles of the people
    data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
    # data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # only works in python 3
    data['Title'] = data['Title'].replace(['Lady', 'Countess','the Countess', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    # convert the title to a numerical ID and fill in empty rows
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

    return data


# Create a new feature which classifies people based on their title, alternative to above
# with the categories "Officer": 1, "Royalty": 2, "Mrs": 3, "Miss": 4, "Mr": 5, "Master": 6
def createTitleAlt(data):
    data['Title_alt'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
    # data['Title_alt'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) # only works in python 3
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
        "Countess": "Royalty",
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
    data['Title_alt'] = data.Title_alt.map(Title_Dictionary)

    # convert the title to a numerical ID and fill in empty rows
    title_mapping = {"Officer": 1, "Royalty": 2, "Mrs": 3, "Miss": 4, "Mr": 5, "Master": 6}
    data['Title_alt'] = data['Title_alt'].map(title_mapping)
    data['Title_alt'] = data['Title_alt'].fillna(0)

    return data

def engineer_features(data):
    print ("Engineering features..")

    # Put features in bins with a numerical ID, to make it easier to train on
    data = binAge(data)
    data = binFares(data)

    # Create new features by combining existing ones
    data = combineAgePClass(data)

    # Engineer completely new features
    data = createTicketFirstChar(data)
    data = createFamilySize(data)
    data = createFamilySizeBinned(data)
    data = createIsAlone(data)
    data = createTitle(data)
    data = createTitleAlt(data)

    return data


def gridSearchRandomForest():
    clf_raw = RandomForestClassifier()
    param_grid = {'max_features': [1, int(np.sqrt(len(self.featureList))), len(self.featureList)],
                    'max_depth': [3, None],
                    'min_samples_split' :[2, 3, 10],
                    'min_samples_leaf' : [1, 3, 10],
                    'criterion':['gini', 'entropy'],
                    'bootstrap':[True, False]}

    clf = GridSearchCV(clf_raw, param_grid=param_grid, cv=10)
    clf.fit(train_X, train_Y)

    print("Best parameters:")
    print(clf.best_params_)
    print("Best average accuracy of K-folds of best model:", clf.best_score_)


def evaluate(train_X, train_Y, test):
    # Evaluation
    runs = 100  # Do it over multiple runs to get estimate of "true" accuracy distribution
    accus = np.zeros((runs, 1))

    print("Using variables:", list(train_X))

    print("Evaluating model performance..")

    for run in range(runs):
        # generate a random train / test split. Corresponds to 10-fold cv
        cv_X_train, cv_X_test, cv_Y_train, cv_Y_test = train_test_split(train_X, train_Y, test_size=0.1)

        # decision tree -> 78.0 acc
        # model = DecisionTreeClassifier()
        # model.fit(cv_X_train, cv_Y_train)
        # # calc mean accuracy over predicted test set
        # accus[run] = model.score(cv_X_test, cv_Y_test)

        # random forest -> 83.5 acc
        model = RandomForestClassifier(bootstrap=randomForestParams['bootstrap'], min_samples_leaf=randomForestParams['min_samples_leaf'], min_samples_split=randomForestParams['min_samples_leaf'], criterion=randomForestParams['criterion'], max_features=randomForestParams['max_features'], max_depth=randomForestParams['max_depth'], n_estimators=250)
        model.fit(cv_X_train, cv_Y_train)
        # calc mean accuracy over predicted test set
        accus[run] = model.score(cv_X_test, cv_Y_test)


    print("Mean test accuracy:", str(np.mean(accus)), "| Std:", str(np.std(accus)) + ". in", str(runs), "runs.")


def train(train_X, train_Y):
    print("Training model on all data..")
    model = RandomForestClassifier(bootstrap=randomForestParams['bootstrap'], min_samples_leaf=randomForestParams['min_samples_leaf'], min_samples_split=randomForestParams['min_samples_leaf'], criterion=randomForestParams['criterion'], max_features=randomForestParams['max_features'], max_depth=randomForestParams['max_depth'], n_estimators=250)
    model.fit(train_X, train_Y)
    return model


def makePredictions(model, test, labels):
    # save predictions as PassengerId, survived
    test = np.array(test)
    y_pred = model.predict(test)
    predictions = []
    for i, prediction in enumerate(y_pred):
        predictions.append([labels[i], prediction])

    # write to file
    df = pd.DataFrame(predictions)
    df.columns = ['PassengerId', 'Survived']
    df.to_csv("tj_test.csv", index=False)

    print ("Saved predictions")

def main():
    # read in data from csv files
    train_df = pd.read_csv('../Data/train.csv')
    test_df = pd.read_csv('../Data/test.csv')

    # preview the data
    train_df.head()

    # Remove survived
    train_X = train_df
    train_Y = train_X["Survived"]
    del train_X["Survived"] # Delete because it is numeric and we want to index independent numerical

    # preview the data
    print (train_X.head())

    train_X = preprocess(train_X)
    test_df = preprocess(test_df)

    train_X = engineer_features(train_X)
    test_df = engineer_features(test_df)

    print (train_X.head())

    # save labels of test set
    labels = test_df['PassengerId']

    # drop non numeric variables
    train_X = train_X.filter(items=(train_X.dtypes[train_X.dtypes != 'object'].index))
    test_df = test_df.filter(items=(test_df.dtypes[test_df.dtypes != 'object'].index))

    # drop some other features
    train_X = train_X.drop(['PassengerId','SibSp','Parch'], axis=1)
    test_df = test_df.drop(['PassengerId','SibSp','Parch'], axis=1)

    # get a performance indication
    evaluate(train_X, train_Y, test_df)

    # train model on complete dataset
    model = train(train_X, train_Y)

    # predict for test file and save
    makePredictions(model, test_df, labels)


if __name__ == '__main__':
    main()
