import pandas as pd
from IO import Saver

class TitanicSaver(Saver):
    def save_predictions(self, predictions, file_name):
        df = pd.DataFrame(predictions)
        df.columns = ['PassengerId', 'Survived']
        df.to_csv(file_name, index=False)