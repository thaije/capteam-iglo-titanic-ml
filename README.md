# Capteam Iglo - Towards a Unitary Titanic Death Determinant Theory
Code for Kaggle competition Titanic: Machine Learning from Disaster


## How to import and run a Notebook
- In this folder you have the Tjalling_notebook.ipynb file, you can import this in a kernel on Keggle.
- Go to https://www.kaggle.com/c/titanic/kernels
- Click new Kernel > Notebook
- At the top press the cloud icon with the arrow, and upload the .ipynb file.
- Run each codepart from top to bottom by clicking in the code, and pressing the play button.



## Feature interpretations
### General
Below some points which were found by running some models (RF, SVM, GRBT, all with Gridsearch) with various features selected.
- If a feature depicts categories, or the values are not a linear sequence or something -> Use one-hot encoding. E.g. the Title feature: Mrs category (ID 0) compared to the Master category (ID 1) is not better, worse, or logically seen before or after the Master category (with ID 1), so it is better to split the categories into different features with one-hot encoding.
- If the feature depicts a linear increase/decrease in something, or it is a logical sequence, one-hot encoding (or binning?) seems unnecessary. E.g. FamilySize doesn't need one-hot encoding.
- To capture interplay between features, some features can be binned (such as Age) to combine them with other features (such as Age*PClass), which may give a increased performance.
- Simplifying features which indirectly capture the same feature to one feature can give better results by reducing the bias towards this certain feature, e.g.: FamilySize, FamilySize_cat, SibSp, Parch, IsAlone. These features can all be captured in the single feature FamilySize.
- Outlier detection seems to help increase performance (atleast for KNN)

### Model / feature specific
- As KNN is of limited complexity it does not cope well with too much features.
- KNN seems to work much best with binned features, E.g. replacing Age_cat, Fare_cat, etc for their non-binned version reduces test set performance by ~10%.
- Bayes seems similar to KNN, doesn't like too many features
- MLP, ExtraTrees seem to like as many features as you can throw at it, the more (good features) the better
- One-hot encoding the Decks seems to lead to overfitting -> Train score higher, test score lower (tested on RF, SVM, GRBT). Probably because of lots of missing data
- Same problem for Namelength as with Deck


This week:
Points:
- tweak ensemble, add weights
- ensemble is shitty, it just redoes a fit on all data.  Feature selection has no use. 1 model with feature selection has other scores that its ensemble.
- feature selection for models
- model selection for ensembles
- do we know how the models work? feature selection for some?
- Alternative for CV, Stratified CV? Better train score seems to indicate overfitting, as test score gets lower.
