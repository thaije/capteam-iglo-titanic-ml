import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from IO import TitanicLoader

class LuukLoader(TitanicLoader):
    def preprocess_data(self, data):
        # Remove columns 'PassengerId' and 'Survived' from the data sets.
        data.drop(['PassengerId'], axis=1)
        # Get rid of NaN values
        data['Cabin'] = data['Cabin'].fillna('None')
        data['Embarked'] = data['Embarked'].fillna('None')
        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
        data['Age'] = data['Age'].fillna(data['Age'].mean())
        # Handle string variables (each unique string value represented as an integer)
        features = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        for feature in features:
            feature_values = list(set().union(data[feature].unique()))
            le = preprocessing.LabelEncoder()
            le.fit(feature_values)
            data[feature] = le.transform(data[feature])

        # Convert dataframes to numpy arrays.
        return data


