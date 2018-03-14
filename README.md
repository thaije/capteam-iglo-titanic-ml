# Capteam Iglo - Towards a Unitary Titanic Death Determinant Theory
Code for Kaggle competition Titanic: Machine Learning from Disaster


## How to import and run a Notebook
- In this folder you have the Tjalling_notebook.ipynb file, you can import this in a kernel on Keggle.
- Go to https://www.kaggle.com/c/titanic/kernels
- Click new Kernel > Notebook
- At the top press the cloud icon with the arrow, and upload the .ipynb file.
- Run each codepart from top to bottom by clicking in the code, and pressing the play button.



Todo:
Fixes:
+ train and test together before feature selection - K
+ remove passenger ID - K

Models:
+ XGBoost - Lis.
+ implement (a lot of) models - 2 for L (ensemble random forrest stuff) and S (naive bayes, nn, GBRT)
+ Outlier detection - K
+ voting ensemble learning - T
+ Check GP? - Z

Feature engineering
- fill missing data by prediction based on other features: https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
+ One-hot encoding: (titles)
- ticket prefix
- try something with cabin
- predict nationality
