# ScrollSense

Built a machine learning system that analyzes short-video usage behavior and predicts users' usage risk levels.

The project combines survey-based data collection, exploratory data analysis, machine learning, deep learning, hyperparameter tuning, and Explainable AI (SHAP) in order to investigate patterns associated with short-video usage.

---

## Project Overview

Short-form video platforms have become an important part of everyday digital consumption. While they can provide entertainment and useful content, the excessive usage may be associated with difficulties like difficulty controlling usage.

ScrollSense was developed to explore whether behavioral and usage-related factors can be used to classify users into different short-video usage risk levels.

The survey was initially shared with family and friends, and additional responses are being collected to expand the dataset and improve future versions of the project.

---

## Data Collection

A survey was designed to collect information about participants' short-video usage patterns.

The collected features include:

- Age.
- Gender.
- Occupation.
- Platforms used.
- Daily short-video usage.
- Daily application opens.
- Typical usage time.
- Main reason for using short-video platforms.
- Average session duration.

The survey also included behavioral questions related to:

- Difficulty stopping.
- Watching more than planned.
- Unintentional application opening.
- Attempts to reduce usage.
- Impact on study or work.
- Sleep delay.
- Task procrastination.
- Regret after usage.

The behavioral responses were converted from a Likert scale into numerical values and combined to calculate an overall risk score.

The risk score was then divided into three categories:

-  **Low Risk**
-  **Moderate Risk**
-  **High Risk**

---

## Exploratory Data Analysis

The project includes exploratory analysis to investigate relationships between short-video usage patterns and risk scores.

Examples include:

- Risk score distribution
- Daily usage vs. risk score
- Session duration vs. risk score
- Typical usage time vs. risk score
- Average risk score across usage categories

Visualization was performed using:

- Matplotlib
- Seaborn

---

## Machine Learning

Several classification models were developed and compared:

### Logistic Regression
Used as a baseline classification model.

### Random Forest
Used to capture nonlinear relationships between user characteristics and risk levels.

### XGBoost
Used as a gradient boosting approach for classification.

### Deep Learning
A neural network was developed using TensorFlow/Keras.

The models were evaluated using:

- Accuracy.
- Macro F1.
- Weighted .
- Stratified 5-Fold Cross-Validation.

Because the dataset is relatively small and the classes are not perfectly balanced, Macro F1 was also considered when comparing model performance.

---

## Hyperparameter Tuning

Grid Search with 5-fold cross-validation was used to tune the XGBoost model.

The search explored different combinations of:

- Learning rate.
- Maximum tree depth.
- Number of estimators.

The objective was to identify a better-performing configuration while maintaining a proper separation between training and test data.

---

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to make the model's predictions more interpretable.

The SHAP analysis helps identify which transformed features contribute most to the model's predictions.

This provides insight into **why** the model makes certain predictions rather than treating the model as a complete black box.

---

## Streamlit Application

A Streamlit application was developed to provide an interactive interface for the project.

The application allows users to enter their short-video usage information and receive a predicted risk level.

The application also uses the trained machine learning model and stored analysis results.

---

## 🛠️ Technologies Used

- Python.
- Pandas.
- NumPy.
- Matplotlib.
- Seaborn.
- Scikit-learn.
- XGBoost.
- TensorFlow / Keras.
- SHAP.
- Streamlit.
- Joblib.

---

## Results

Multiple machine learning approaches were evaluated, including Logistic Regression, Random Forest, XGBoost, and a Deep Learning neural network.

The deep learning model was included to explore a more complex modeling approach, but it did not achieve the best performance on the available dataset. Given the relatively small number of survey responses, traditional machine learning approaches were more suitable for the current version of the project.

Random Forest was selected as the final model based on its overall performance across the evaluation metrics.

As more survey responses are collected, the dataset can be expanded and the deep learning approach can be reevaluated in future iterations.

---

## Project Structure

```text
ScrollSense/
│
├── app.py
├── main.py
├── survey.csv
│
├── cv_results.csv
├── deep_learning_cv_results.csv
├── final_model_results.csv
├── shap_feature_importance.csv
│
├── random_forest_model.pkl
│
├── .gitignore
└── README.md
