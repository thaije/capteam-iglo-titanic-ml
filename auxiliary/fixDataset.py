import pandas as pd
import numpy as np


kaggle_test_file = "../Data/test.csv"
full_dataset = "../Data/fullData/titanic_dataset.csv"
output_file = "../Data/fullData/test_complete.csv"


output = pd.read_csv(kaggle_test_file)
# create new Survived column initialized to -1
output['Survived'] = pd.Series( np.array(list([-1] * len(output))), index=output.index )

# clean the datasets by removing excess "" characters from the name fields which complicates matching
kaggle_test_cleaned = pd.read_csv(kaggle_test_file).replace('\"', '', regex=True)
full_test_cleaned = pd.read_csv(full_dataset).replace('\"', '', regex=True)


print ("Matching passengers")

# Match every person in the kaggle test set with someone from the full data set,
# based on their name, and age if necessary
divergents = 0
for index, row in kaggle_test_cleaned.iterrows():

    # try to find person with the same name
    matching_passengers = full_test_cleaned[ (full_test_cleaned['Name'] == row['Name'])  ]

    # if there are multiple, also filter on age
    if len(matching_passengers) > 1:
        matching_passengers = full_test_cleaned[ (full_test_cleaned['Name'] == row['Name']) & (full_test_cleaned['Age'] == row['Age']) ]

    if len(matching_passengers) != 1:
        print("Found a divergent number of matches (%d) for passenger %s" % (len(matching_passengers), row['Name']) )
        divergents += 1
    else:
        # copy Survived value from complete dataset to Kaggle untouched dataset
        survived = np.array(matching_passengers)[0][1]
        output.at[index, 'Survived'] = survived

print ("\nFound %d divergent passengers who have more or less than 1 match" % divergents)

print(output.head())

output.to_csv(output_file, index=False)
print ("Output written to %s" % output_file)
