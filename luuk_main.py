import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import preprocessing, svm
from sklearn.neural_network import MLPClassifier

import CrossValidate as CV

if __name__ == '__main__':
    train = pd.read_csv('Data\\train.csv')
    test = pd.read_csv('Data\\test.csv')

    # Get the true class labels from the training set.
    y_train = np.array(train['Survived'])
    
    # Remove columns 'PassengerId' and 'Survived' from the data sets.
    X_train = train.drop(['PassengerId', 'Survived'], axis=1)
    X_test = test.drop(['PassengerId'], axis=1)

    # Get rid of NaN values
    X_train['Cabin'] = X_train['Cabin'].fillna('None')
    X_train['Embarked'] = X_train['Embarked'].fillna('None')
    X_train['Fare'] = X_train['Fare'].fillna(X_train['Fare'].mean())
    X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())

    X_test['Cabin'] = X_test['Cabin'].fillna('None')
    X_test['Embarked'] = X_test['Embarked'].fillna('None')
    X_test['Fare'] = X_train['Fare'].fillna(X_test['Fare'].mean())
    X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean())

    # Handle string variables (each unique string value represented as an integer)
    features = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    for feature in features:
        feature_values = list(set().union(X_train[feature].unique(), X_test[feature].unique()))
        le = preprocessing.LabelEncoder()
        le.fit(feature_values)  
        X_train[feature] = le.transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

    # Convert dataframes to numpy arrays.
    X_train = pd.DataFrame.as_matrix(X_train)
    X_test = pd.DataFrame.as_matrix(X_test)

    # Model Selection
    clf = RandomForestClassifier()
    #clf = AdaBoostClassifier()
    #clf = svm.SVC()
    #clf = MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(100,2))

    # Cross Validation
    clf = CV.KFold(X_train, y_train, clf, 2)
    #clf = CV.RepeatedKFold(X_train, y_train, clf, 10, 10)
    #clf = CV.LeaveOneOut(X_train, y_train, clf)
    #clf = CV.StratifiedKFold(X_train, y_train, clf, 10)
    
    # Prediction
    y_pred = clf.predict(X_test)

    # Write predictions to csv file
    passenger_ids = list(train['PassengerId'])
    id_offset = len(X_train)
    predictions = []
    for i, prediction in enumerate(y_pred):
        predictions.append([passenger_ids[i] + id_offset, prediction])
    df = pd.DataFrame(predictions)
    df.columns = ['PassengerId', 'Survived']
    df.to_csv('submission.csv', index=False)
    
