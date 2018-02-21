import pandas as pd
import numpy as np

import preprocessing.preprocess as pp
import featureEngineering.engineerFeatures as fe

##################
# NOTE: Python 3 #
##################

pathTrainSet = "data/train.csv"
pathTestSet = "data/test.csv"



# NOTE: Python 3
def main():
    # read in data
    train_df, test_df = readInData()

    ## Preprocessing
    train_df = pp.preprocess(train_df)

    ## Feature engineering
    # bin features into categories
    train_df = fe.binExistingFeatures(train_df)

    # create new features by combining others
    train_df = fe.createCombinedFeatures(train_df)

    # engineer some new features
    train_df = fe.createNewFeatures(train_df)

    # split the data
    train_X = train_df
    train_Y = train_X["Survived"]
    del train_X["Survived"]

    ## Feature selection
    train_X = featureSelection(train_X)

    ## Evaluation
    evaluate(train_X, train_Y)



def readInData():
    train_df = pd.read_csv( pathTrainSet )
    test_df = pd.read_csv( pathTestSet )

    print( "\n" + ('-' * 40) )
    print( " Original Data")
    print( '-' * 40)
    print( train_df.head() )

    return train_df, test_df


# TODO: Select the most predictive features using scikit learn or something
def featureSelection(data):
    
    # return data.drop(['Age*Class','SibSp','Parch'], axis=1)
    return data


# TODO: train model(s) on the data using CV
# TODO: try ensemble learning
def evaluate(train_X, train_Y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    runs = 100  # Do it over multiple runs to get estimate of "true" accuracy distribution
    accus = np.zeros((runs, 1))

    # Only numerical attributes are used
    numeric_variables = list(train_X.dtypes[train_X.dtypes != 'object'].index)
    print("Using variables:", numeric_variables)

    print("Training model..")
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

    print("Mean test accuracy:", str(np.mean(accus)), "| Std:", str(np.std(accus)) + ". in", str(runs), "runs.")




if __name__ == '__main__':
    main()
