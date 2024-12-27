# Diabetes Prediction Project

This project focuses on predicting diabetes using machine learning algorithms. The dataset used contains various health-related features that are analyzed to determine the likelihood of diabetes in patients.

## Project Overview

The Diabetes Prediction project utilizes a dataset containing information about patients, including features such as pregnancies, glucose levels, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. The goal is to build and compare different machine learning models to predict the likelihood of diabetes based on these features.

## Data Preprocessing

The dataset undergoes several preprocessing steps:
- Loading and initial exploration of the data
- Handling missing values
- Feature engineering (e.g., creating new features like Glucose_Level, BMI_Status, and Age_Group)
- Data visualization for insights

## Models Implemented

The project implements and compares the following machine learning models:
1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest

## Model Evaluation

Each model is evaluated based on its performance on both training and test datasets. The evaluation metrics include:
- Train Accuracy
- Test Accuracy

## Hyperparameter Tuning

The project includes hyperparameter tuning for each model to optimize their performance. The best hyperparameters for each model are reported.

## Feature Importance

For tree-based models (Decision Tree and Random Forest), feature importance is analyzed to understand which factors contribute most to the prediction of diabetes.

## Results

The final results show that the Random Forest model performs best with a test accuracy of 82.28% after hyperparameter tuning. The SVM model also shows good performance with a test accuracy of 81.01%.

## Technologies Used

- Python
- Pandas for data manipulation
- Numpy for numerical operations
- Seaborn and Matplotlib for data visualization
- Scikit-learn for machine learning models and evaluation

## Future Work

Potential areas for improvement and expansion of the project include:
- Exploring more advanced feature engineering techniques
- Implementing ensemble methods
- Investigating other machine learning algorithms
- Collecting more data to improve model performance
