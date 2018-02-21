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
    print( " Data After putting features in bins")
    print( '-' * 40)
    print( data.head() )

    return data


# combine features to potentially capture correlations
def createCombinedFeatures(data):

    # Creat eew class Pclass*Age
    data['Age*Class'] = data.Age_cat * data.Pclass
    data.loc[:, ['Age*Class', 'Age_cat', 'Pclass']].head(10)

    print( "\n" + ('-' * 40) )
    print( " Data After creating new features by combination")
    print( '-' * 40)
    print( data.head() )

    return data

def createNewFeatures(data):

    # Create a new feature from the first character of the Ticket
    data['Ticket_firstchar'] = data['Ticket']
    data.Ticket_firstchar = data.Ticket_firstchar.map(lambda x: x[0])
    data["Ticket_firstchar"].replace(['A', 'P', 'S', 'C', 'W', 'F', 'L', '1','2','3','4','5','6','7','8','9'], [10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9], inplace=True) # To numerical values

    # Create new feature family = self + sibelings/spouse + Parents/children
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # create new feature which classifies people as alone or not
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    

    print( "\n" + ('-' * 40) )
    print( " Data After engineering new features")
    print( '-' * 40)
    print( data.head() )

    return data
