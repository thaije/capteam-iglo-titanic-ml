# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# data analysis and wrangling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from auxiliary import feature_importance as fs
from sklearn.feature_selection import chi2
# visualization


# machine learning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# read in data from csv files
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# preview the data
train_df.head()

### PREPROCESSING

# Remove survived
train_X = train_df
train_Y = train_X["Survived"]
del train_X["Survived"] # Delete because it is numeric and we want to index independent numerical

# preview the data
train_X.head()


# Replace sex to numeric
# replace sex attribute to numerical: female=0, male=1
train_X['Sex'].replace(['female','male'], [0,1],inplace=True)

# preview the data
train_X.head()

# Make age categories
train_X["Age"].fillna(train_X.Age.mean(), inplace=True)

# Age to category of age
interval = (0, 5, 12, 18, 25, 35, 60, 120)
cats = [0,1,2,3,4,5,6]
train_X["Age_cat"] = pd.cut(train_X.Age, interval, labels=cats)
train_X["Age_cat"].head()

#Categorize embarked
freq_port = train_X.Embarked.dropna().mode()[0]
print ("Replacing NaN values of embarked with most frequent port:", freq_port)
train_X['Embarked'] = train_X['Embarked'].fillna(freq_port)

# we have three types of ports
print(pd.unique(train_X['Embarked']))

# replace port numbers with category ID
train_X["Embarked_cat"] = train_X["Embarked"]
train_X["Embarked_cat"].replace(['S','C','Q'], [1,2,3], inplace=True) # To numerical values
print(pd.unique(train_X['Embarked_cat']))

# preview the data
train_X.head()


#Fares in numerical categories
# Fill in NaN values
train_X["Fare"].fillna(train_X.Fare.mean(), inplace=True)
train_X.head()

# Put fares into 4 bands (see https://www.kaggle.com/polarhut/titanic-data-science-solutions)
train_X['Fare_cat'] = train_X['Fare']
train_X.loc[ train_X['Fare_cat'] <= 7.91, 'Fare_cat'] = 0
train_X.loc[(train_X['Fare_cat'] > 7.91) & (train_X['Fare'] <= 14.454), 'Fare_cat'] = 1
train_X.loc[(train_X['Fare_cat'] > 14.454) & (train_X['Fare'] <= 31), 'Fare_cat']   = 2
train_X.loc[ train_X['Fare_cat'] > 31, 'Fare_cat'] = 3
train_X['Fare_cat'] = train_X['Fare_cat'].astype(int)


# interval = (0, 7.91, 14.54, 31, 1000)
# cats = [0,1,2,3]
# train_X['Fare_cat'] = train_X['Fare']
# train_X['Fare_cat'] = pd.cut(train_X.Fare, interval, labels=cats)
train_X['Fare_cat'].head()


# Create family size and isalone
# family = self + sibelings/spouse + Parents/childs
train_X['FamilySize'] = train_X['SibSp'] + train_X['Parch'] + 1

train_X['IsAlone'] = 0
train_X.loc[train_X['FamilySize'] == 1, 'IsAlone'] = 1

# preview the data
train_X.head()

# Creat eew class Pclass*Age
train_X['Age*Class'] = train_X.Age_cat * train_X.Pclass
train_X.loc[:, ['Age*Class', 'Age_cat', 'Pclass']].head(10)


train_X.Ticket = train_X.Ticket.map(lambda x: x[0]) # This transforms Ticket to features of ticket number
train_X["Ticket"].replace(['A', 'P', 'S', 'C', 'W', 'F', 'L', '1','2','3','4','5','6','7','8','9'], [10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9], inplace=True) # To numerical values


# remove
train_X = train_X.drop(['PassengerId','SibSp','Parch'], axis=1)

# preview the data
train_X.head()

# Evaluation
runs = 100  # Do it over multiple runs to get estimate of "true" accuracy distribution
accus = np.zeros((runs, 1))

# Only numerical attributes are used
numeric_variables = list(train_X.dtypes[train_X.dtypes != 'object'].index)
print("Using variables:", numeric_variables)

print("Running..")

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

clf = GridSearchCV(SVC(), param_grid, cv=5,
                   scoring='accuracy')
clf.fit(X_train, y_train)

for run in range(runs):
    # generate a random train / test split
    cv_X_train, cv_X_test, cv_Y_train, cv_Y_test = train_test_split(train_X, train_Y, test_size=0.3)

    # decision tree -> 80.0 acc
    #     model = DecisionTreeClassifier()
    #     model.fit(cv_X_train[numeric_variables], cv_Y_train)
    #     # calc mean accuracy over predicted test set
    #     accus[run] = model.score(cv_X_test[numeric_variables], cv_Y_test)

    # random forest -> 81.2 acc
    model = RandomForestClassifier(n_estimators=250)
    model.fit(cv_X_train[numeric_variables], cv_Y_train)
    # calc mean accuracy over predicted test set
    accus[run] = model.score(cv_X_test[numeric_variables], cv_Y_test)
    # accus[run] = accuracy_score(cv_Y_test, model.predict(cv_X_test[numeric_variables]))
    if run == 0:
        fs.analyze_feature_importance(model, numeric_variables)
        chi = fs.obtain_feature_scores(chi2, cv_X_test[numeric_variables], cv_Y_test, numeric_variables)

print("Mean test accuracy:", str(np.mean(accus)), "| Std:", str(np.std(accus)) + ". in", str(runs), "runs.")
