import pandas as pd
from input_output.Saver import Saver
from joblib import dump

# Saver specific for the Titanic task
# The Saver saves the predictions to an submission file
class TitanicSaver(Saver):
    def save_predictions(self, predictions, file_name):
        df = pd.DataFrame(predictions)
        df.columns = ['PassengerId', 'Survived']
        df.to_csv(file_name, index=False)

        print ("Predictions written to " , file_name)

    def save_model(self, model, file_name):
        dump(model, file_name) 