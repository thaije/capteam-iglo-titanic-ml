import pandas as pd
import numpy as np

##################
# NOTE: Python 3 #
##################

# train_X = data
# train_Y = train_X["Survived"]


# Put features in bins with a numerical ID, to make it easier to train on
def binExistingFeatures(data):

    # Bin Age and save it as a new feature
    interval = (0, 5, 12, 18, 25, 35, 60, 120)
    cats = [0,1,2,3,4,5,6]
    data["Age_cat"] = pd.cut(data.Age, interval, labels=cats)


    # Put fares into 4 bins (see https://www.kaggle.com/polarhut/titanic-data-science-solutions)
    data['Fare_cat'] = data['Fare']
    data.loc[ data['Fare_cat'] <= 7.91, 'Fare_cat'] = 0
    data.loc[(data['Fare_cat'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
    data.loc[(data['Fare_cat'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat']   = 2
    data.loc[ data['Fare_cat'] > 31, 'Fare_cat'] = 3
    data['Fare_cat'] = data['Fare_cat'].astype(int)


    print( "\n" + ('-' * 40) )
    print( " Data after putting features in bins")
    print( '-' * 40)
    print( data.head() )

    return data


# combine features to potentially capture correlations
def createCombinedFeatures(data):

    # Creat eew class Pclass*Age
    data['Age*Class'] = data.Age_cat * data.Pclass
    data.loc[:, ['Age*Class', 'Age_cat', 'Pclass']].head(10)

    print( "\n" + ('-' * 40) )
    print( " Data after creating new features by combination")
    print( '-' * 40)
    print( data.head() )

    return data

def createNewFeatures(data):

    ## Create a new feature from the first character of the Ticket
    data['Ticket_firstchar'] = data['Ticket']
    data.Ticket_firstchar = data.Ticket_firstchar.map(lambda x: x[0])
    data["Ticket_firstchar"].replace(['A', 'P', 'S', 'C', 'W', 'F', 'L', '1','2','3','4','5','6','7','8','9'], [10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9], inplace=True)


    ## Create new feature family = self + sibelings/spouse + Parents/children
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1


    ## create new feature based on family size, but in categories
    data['FamilySize_cat'] = data['FamilySize']
    data.loc[ data['FamilySize_cat'] <= 1, 'FamilySize_cat'] = 1
    data.loc[(data['FamilySize_cat'] >= 2) & (data['FamilySize_cat'] < 5), 'FamilySize_cat'] = 2
    data.loc[data['FamilySize_cat'] >= 5 , 'FamilySize_cat']   = 3
    # print ( data.groupby('FamilySize_cat')['PassengerId'].nunique() )


    ## create new feature which classifies people as alone or not
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1


    ## create a new feature which classifies people based on their title,
    ## with categories "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5
    # first extract the titles of the people
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','the Countess', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    # convert the title to a numerical ID and fill in empty rows
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)


    ## create a new feature which classifies people based on their title, alternative to above
    ## with the categories "Officer": 1, "Royalty": 2, "Mrs": 3, "Miss": 4, "Mr": 5, "Master": 6
    data['Title_alt'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
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


    print( "\n" + ('-' * 40) )
    print( " Data after engineering new features")
    print( '-' * 40)
    print( data.head() )

    return data
