# Income Prediction Data Report

This report analyzes classification methods applied to the US Adult dataset, aiming to predict whether a given adult makes more than $50,000 annually.

## Features of the Dataset

- **age**: Age of an individual
- **workclass**: Employment status of an individual
- **fnlwgt**: Final weight, representing the number of people the census believes the entry represents (not used in this exercise)
- **education**: Highest level of education achieved by an individual
- **education-num**: Highest level of education achieved in numerical form
- **marital-status**: Marital status of an individual
- **occupation**: General type of occupation of an individual
- **relationship**: Relative position of the individual
- **race**: Description of an individual's race
- **sex**: Biological sex of the individual
- **capital-gain**: Capital gains for an individual
- **capital-loss**: Capital loss for an individual
- **hours-per-week**: Hours an individual has reported to work per week
- **native-country**: Country of origin for an individual
- **label**: Whether an individual makes more than $50,000 annually

## Libraries Used

- Pandas: For machine learning, data science, and data analysis operations
- NumPy: For multi-dimensional arrays, forming the foundation of Pandas
- Matplotlib and Seaborn: For data visualization

## Dataset Overview

Two datasets were provided:
- `adult.data`: Training dataset for the models
- `adult.test`: Unseen data used to test selected trained models

## Initial Data Exploration

- Distribution curve plotted for the age column, showing a skew towards ages 20–50.
- Countplot analysis for education and marital status columns revealing insights into the dataset's demographics.
- Average hours worked per week analyzed by education level and marital status.

## Income Analysis

- High school graduates have the highest number of people earning below $50,000, while bachelor's degree holders have the highest number of people earning above $50,000.
- Men dominate both below and above $50,000 earnings, potentially reflecting gender distribution in the dataset.
- Administrative jobs have the highest number of people earning below $50,000, while managerial roles have the highest number of people earning above $50,000.

## Model Building

- Created age groups and compared them with income levels.
- Selected important features for modeling using RFECV with DecisionTreeClassifier.
- Built three classification models (KNeighborsClassifier, DecisionTreeClassifier, GaussianNB) using RandomizedSearchCV for validation and hyperparameter tuning.
- Saved the best performing Decision Tree Classifier model using the pickle library.

## Ensemble Methods

- Built two ensemble methods (XGBClassifier, RandomForestClassifier) using RandomizedSearchCV for validation and hyperparameter tuning.
- Saved the best performing XGBClassifier model using the pickle library.

## Conclusion

- Decision Tree Classifier performed best among the three classifiers, while XGBClassifier outperformed RandomForestClassifier in the ensemble methods.
- Both ensemble methods demonstrated superior performance compared to the three classification methods.
