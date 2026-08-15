import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV
import shap
import joblib

df= pd.read_csv("survey.csv")

print(df.columns.tolist())

# Transalte columns 
col_map = {
    "العمر": "Age",
    "الجنس": "Gender",
    "المهنة": "Occupation",
    "ما هي المنصات التي تستخدمها؟ (يمكنك اختيار أكثر من إجابة)": "Platforms_Used",
    "كم ساعة تقضي يوميًا في مشاهدة الفيديوهات القصيرة؟": "Daily_Usage",
    "كم مرة تفتح تطبيقات الفيديوهات القصيرة يوميًا؟": "Daily_App_Opens",
    "متى تستخدمها غالبًا؟": "Typical_Usage_Time",
    "ما السبب الرئيسي لاستخدامك للفيديوهات القصيرة؟": "Main_Usage_Reason",
    "كم يبلغ متوسط مدة الجلسة الواحدة لمشاهدة الفيديوهات القصيرة؟": "Average_Session_Duration",
    "إلى أي مدى تنطبق عليك العبارات التالية؟.أجد صعوبة في التوقف عن المشاهدة": "Difficulty_Stopping",
    "إلى أي مدى تنطبق عليك العبارات التالية؟.أشاهد أكثر مما خططت": "Watch_More_Than_Planned",
    "إلى أي مدى تنطبق عليك العبارات التالية؟.أفتح التطبيق دون قصد": "Unintentional_App_Opening",
    "إلى أي مدى تنطبق عليك العبارات التالية؟.أحاول تقليل الاستخدام": "Attempt_To_Reduce",
    "إلى أي مدى تنطبق عليك العبارات التالية؟.تؤثر على دراستي أو عملي": "Impact_On_Study_Work",
    "إلى أي مدى تنطبق عليك العبارات التالية؟.تؤخر نومي": "Sleep_Delay",
    "إلى أي مدى تنطبق عليك العبارات التالية؟.أؤجل مهامي": "Task_Procrastination",
    "إلى أي مدى تنطبق عليك العبارات التالية؟.أشعر بالندم بعد استخدامها": "Regret_After_Use"
}

col_to_drop = ["Id", "Start time", "Completion time", "Email", "Name"]
df= df.drop(columns=col_to_drop)

df= df.rename(columns=col_map)
print(df.columns.tolist())

for col in df.columns:
    print("\n" + col)
    print(df[col].unique())

behavior_cols = [
    "Difficulty_Stopping",
    "Watch_More_Than_Planned",
    "Unintentional_App_Opening",
    "Attempt_To_Reduce",
    "Impact_On_Study_Work",
    "Sleep_Delay",
    "Task_Procrastination",
    "Regret_After_Use"
]

likert_map= {
    "أبدًا":0,
    "نادرًا":1,
    "أحيانًا":2,
    "غالبًا":3,
    "دائمًا":4 
}

for col in behavior_cols:
    df[col]= df[col].map(likert_map)

df["Risk_score"]= df[behavior_cols].sum(axis=1)

print(df.shape)

print("\nMissing values: ", df.isnull().sum())
print("\nDuplicates: ", df.duplicated().sum())

print(df["Age"].value_counts())
print(df["Gender"].value_counts(dropna=False))
print(df["Occupation"].value_counts())

df = df.rename(columns={"Risk_score": "Risk_Score"})

print(df["Risk_Score"].describe())
print(df["Risk_Score"].value_counts().sort_index())

plt.figure(figsize=(8, 5))
plt.hist(df["Risk_Score"], bins=range(0, 34), edgecolor="black")
plt.xlabel("Risk Score")
plt.ylabel("Number of Participants")
plt.title("Distribution of Short-Video Usage Risk Score")

plt.show()

# Daily usage vs Risk Score

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Daily_Usage",
    y="Risk_Score"
)

plt.xlabel("Daily Short-Video Usage")
plt.ylabel("Risk Score")
plt.title("Daily Usage vs Risk Score")
plt.xticks(rotation=30)
plt.show()

#Session Duration vs Risk Score

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Average_Session_Duration",
    y="Risk_Score"
)

plt.xlabel("Average Session Duration")
plt.ylabel("Risk Score")
plt.title("Session Duration vs Risk Score")
plt.xticks(rotation=30)
plt.show()

# Usage Time vs Risk Score 

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Typical_Usage_Time",
    y="Risk_Score"
)

plt.xlabel("Typical Usage Time")
plt.ylabel("Risk Score")
plt.title("Typical Usage Time vs Risk Score")
plt.xticks(rotation=30)
plt.show()

# Calulate avg of risk score for each category

avg_score_daily_usage= df.groupby("Daily_Usage")["Risk_Score"].mean().sort_values(ascending=False)

print(avg_score_daily_usage)

avg_score_session_duration= df.groupby("Average_Session_Duration")["Risk_Score"].mean().sort_values(ascending=False)
print(avg_score_session_duration)

df["Risk_Level"]= pd.cut(df["Risk_Score"],
                        bins=[-1, 10, 21, 32],
                        labels= ["Low", "Moderate", "High"])

print(df["Risk_Level"].value_counts())

feature_cols= [
    "Age",
    "Gender",
    "Occupation",
    "Platforms_Used",
    "Daily_Usage",
    "Daily_App_Opens",
    "Typical_Usage_Time",
    "Main_Usage_Reason",
    "Average_Session_Duration"
]

X= df[feature_cols]
y= df["Risk_Level"]

cat_features= X.columns.tolist()

preproc= ColumnTransformer(
    transformers=[ 
        ("cat",
        OneHotEncoder(handle_unknown="ignore"),
        cat_features)
    ]
)

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2,
                                                    random_state=42, stratify=y) #startfied cross validation will be used to get more reliable results

# Start with baseline : Logistic Regression

logistic_model= Pipeline([
    ("preprocessor", preproc),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

y_log_pred= logistic_model.predict(X_test)

print(classification_report(y_test, y_log_pred, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, y_log_pred)

plt.title("Logistic Regression Confusion Matrix")
plt.show()

rf_model=  Pipeline([
    ("preprocessor", preproc),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

rf_model.fit(X_train, y_train) #Train
y_rf_pred = rf_model.predict(X_test) #Predict 
print(classification_report(y_test, y_rf_pred, zero_division=0))

# Encode labels for XGBOOST 

xgb_label_encoder = LabelEncoder()

y_train_xgb = xgb_label_encoder.fit_transform(y_train)
y_test_xgb = xgb_label_encoder.transform(y_test)

print(xgb_label_encoder.classes_)

xgb_model = Pipeline([
    ("preprocessor", preproc),
    ("classifier", XGBClassifier(n_estimators=100,
                                 max_depth=3,
                                 learning_rate=0.1,
                                 random_state=42,
                                 eval_metric="mlogloss" ))
])

xgb_model.fit(X_train, y_train_xgb)
y_xgb_pred = xgb_model.predict(X_test)
y_xgb_pred_labels = xgb_label_encoder.inverse_transform(y_xgb_pred.astype(int)) #convert prediction to og class names

print(classification_report(y_test, y_xgb_pred_labels, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_test, y_xgb_pred_labels)
plt.title("XGBoost Confusion Matrix")
plt.show()

# Deep learning part
# One hot encode features for Neural Network

dl_preproc = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            X_train.columns
        )
    ]
)

X_train_dl= dl_preproc.fit_transform(X_train)
X_test_dl= dl_preproc.transform(X_test)

dl_label_encoder= LabelEncoder()

y_train_dl= dl_label_encoder.fit_transform(y_train)
y_test_dl= dl_label_encoder.transform(y_test)

print("Classes: ", dl_label_encoder.classes_)

#Build neural netwoek 

dl_model= Sequential([
    Input(shape=(X_train_dl.shape[1],)),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(3, activation="softmax")
])

#compile model

dl_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# used since dataset is small

early_stopping= EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

#train

history= dl_model.fit(
    X_train_dl,
    y_train_dl,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)

# evaluate 

test_loss, test_accuracy = dl_model.evaluate(
    X_test_dl,
    y_test_dl,
    verbose=0
)

print("DL Test Accuracy:", test_accuracy)

y_dl_prob = dl_model.predict(X_test_dl)
y_dl_pred = np.argmax(y_dl_prob, axis=1)

print(classification_report(
    y_test_dl,
    y_dl_pred,
    target_names=dl_label_encoder.classes_,
    zero_division=0
))

ConfusionMatrixDisplay.from_predictions(
    y_test_dl,
    y_dl_pred,
    display_labels=dl_label_encoder.classes_
)

plt.title("Deep Learning Confusion Matrix")
plt.show()

cv= StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted"
}

logistic_cv = cross_validate(
    logistic_model,
    X,
    y,
    cv=cv,
    scoring=scoring
)

rf_cv = cross_validate(
    rf_model,
    X,
    y,
    cv=cv,
    scoring=scoring
)

# encode target for xgboost 

xgb_cv_label_encoder = LabelEncoder()

y_encoded = xgb_cv_label_encoder.fit_transform(y)

xgb_cv_model = Pipeline([
    ("preprocessor", preproc),
    ("classifier", XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    ))
])

xgb_cv = cross_validate(
    xgb_cv_model,
    X,
    y_encoded,
    cv=cv,
    scoring=scoring
)
# Deep learning cross validation

dl_cv_results = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
    print(f"\nFold {fold}")

    # Split data

    X_train_fold= X.iloc[train_idx]
    X_val_fold= X.iloc[val_idx]

    y_train_fold= y.iloc[train_idx]
    y_val_fold= y.iloc[val_idx]

    # One hot encode features

    fold_preproc= OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_fold= fold_preproc.fit_transform(X_train_fold)
    X_val_fold= fold_preproc.transform(X_val_fold)

    # Encode target

    fold_label_encoder= LabelEncoder()

    y_train_fold= fold_label_encoder.fit_transform(y_train_fold)
    y_val_fold= fold_label_encoder.transform(y_val_fold)

    # Build neural network

    fold_model= Sequential([
        Input(shape=(X_train_fold.shape[1],)),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")
    ])

    # Compile model

    fold_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Early stopping

    fold_early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Train

    fold_model.fit(
        X_train_fold,
        y_train_fold,
        validation_split=0.2,
        epochs=100,
        batch_size=8,
        callbacks=[fold_early_stopping],
        verbose=0
    )

    # Predict

    y_fold_prob= fold_model.predict(
        X_val_fold,
        verbose=0
    )

    y_fold_pred= np.argmax(
        y_fold_prob,
        axis=1
    )

    # Calculate metrics

    fold_accuracy= accuracy_score(
        y_val_fold,
        y_fold_pred
    )

    fold_macro_f1= f1_score(
        y_val_fold,
        y_fold_pred,
        average="macro"
    )

    fold_weighted_f1 = f1_score(
        y_val_fold,
        y_fold_pred,
        average="weighted"
    )

    dl_cv_results.append({
        "Fold": fold,
        "Accuracy": fold_accuracy,
        "Macro F1": fold_macro_f1,
        "Weighted F1": fold_weighted_f1
    })

print("\nDeep Learning CV Results:")
dl_cv_results = pd.DataFrame(dl_cv_results)
print(dl_cv_results)

# Calculate mean and standard deviation

dl_mean = dl_cv_results[["Accuracy", "Macro F1", "Weighted F1"]].mean()
dl_std = dl_cv_results[["Accuracy", "Macro F1", "Weighted F1"]].std()

print("\nDeep Learning Mean:")
print(dl_mean)

print("\nDeep Learning Standard Deviation:")
print(dl_std)

# comparision table 

cv_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Deep Learning"
    ],

    "Accuracy": [
        logistic_cv["test_accuracy"].mean(),
        rf_cv["test_accuracy"].mean(),
        xgb_cv["test_accuracy"].mean(),
        dl_mean["Accuracy"]
    ],

    "Macro F1": [
        logistic_cv["test_macro_f1"].mean(),
        rf_cv["test_macro_f1"].mean(),
        xgb_cv["test_macro_f1"].mean(),
        dl_mean["Macro F1"]
    ],

    "Weighted F1": [
        logistic_cv["test_weighted_f1"].mean(),
        rf_cv["test_weighted_f1"].mean(),
        xgb_cv["test_weighted_f1"].mean(),
        dl_mean["Weighted F1"]
    ],

    "Macro F1 Std": [
        logistic_cv["test_macro_f1"].std(),
        rf_cv["test_macro_f1"].std(),
        xgb_cv["test_macro_f1"].std(),
        dl_std["Macro F1"]
    ]
})

print("\nFinal CV Comparison:")
print(cv_results)

# XGBoost Hyperparameter Tuning

xgb_tuning_model = Pipeline([
    ("preprocessor", preproc),
    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="mlogloss"
    ))
])

xgb_param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [2, 3, 4],
    "classifier__learning_rate": [0.05, 0.1, 0.2]
}

# Tune ONLY on the training data

xgb_grid_search = GridSearchCV(
    estimator=xgb_tuning_model,
    param_grid=xgb_param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

xgb_grid_search.fit(
    X_train,
    y_train_xgb
)

print("\nBest XGBoost Parameters:")
print(xgb_grid_search.best_params_)

print("\nBest XGBoost CV Macro F1:")
print(xgb_grid_search.best_score_)

# Final evaluation on untouched test set

best_xgb_model = xgb_grid_search.best_estimator_

y_tuned_xgb_pred = best_xgb_model.predict(X_test)

y_tuned_xgb_pred_labels = (
    xgb_label_encoder.inverse_transform(
        y_tuned_xgb_pred.astype(int)
    )
)

print("\nFinal Tuned XGBoost Classification Report:")

print(classification_report(
        y_test,
        y_tuned_xgb_pred_labels,
        zero_division=0)
)

ConfusionMatrixDisplay.from_predictions(y_test, y_tuned_xgb_pred_labels)

plt.title("Final Tuned XGBoost Confusion Matrix")
plt.show()

# Final test metrics

final_accuracy = accuracy_score(y_test, y_tuned_xgb_pred_labels)

final_macro_f1 = f1_score(y_test, y_tuned_xgb_pred_labels, average="macro")

final_weighted_f1 = f1_score(y_test, y_tuned_xgb_pred_labels, average="weighted")

print("\nFinal Test Results:")
print("Accuracy:", final_accuracy)
print("Macro F1:", final_macro_f1)
print("Weighted F1:", final_weighted_f1)

# Final Test Model Comparison

final_model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Deep Learning",
        "Tuned XGBoost"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_log_pred),
        accuracy_score(y_test, y_rf_pred),
        accuracy_score(y_test, y_xgb_pred_labels),
        accuracy_score(y_test_dl, y_dl_pred),
        final_accuracy
    ],

    "Macro F1": [
        f1_score(y_test, y_log_pred, average="macro"),
        f1_score(y_test, y_rf_pred, average="macro"),
        f1_score(y_test, y_xgb_pred_labels, average="macro"),
        f1_score(y_test_dl, y_dl_pred, average="macro"),
        final_macro_f1
    ],

    "Weighted F1": [
        f1_score(y_test, y_log_pred, average="weighted"),
        f1_score(y_test, y_rf_pred, average="weighted"),
        f1_score(y_test, y_xgb_pred_labels, average="weighted"),
        f1_score(y_test_dl, y_dl_pred, average="weighted"),
        final_weighted_f1
    ]
})

print("Final test model comparison ")
print(final_model_results)

# Best model based on Macro F1

best_model_row = final_model_results.loc[final_model_results["Macro F1"].idxmax()]
print("\nBest Final Model:")
print(best_model_row)

# SHAP Explainability for Random Forest

# Get the trained preprocessor
rf_preprocessor = rf_model.named_steps["preprocessor"]

# Get the trained Random Forest classifier
rf_classifier = rf_model.named_steps["classifier"]

# Transform the test data
X_test_rf_transformed = rf_preprocessor.transform(X_test)

# Convert to dense numerical array for SHAP
if hasattr(X_test_rf_transformed, "toarray"):
    X_test_rf_transformed = X_test_rf_transformed.toarray()

X_test_rf_transformed = np.asarray(
    X_test_rf_transformed,
    dtype=np.float64
)

# Get transformed feature names
rf_feature_names = rf_preprocessor.get_feature_names_out()

print("\nSHAP Analysis")
print("Transformed data shape:", X_test_rf_transformed.shape)
print("Number of features:", len(rf_feature_names))

# Create SHAP explainer
rf_explainer = shap.TreeExplainer(rf_classifier)

# Calculate SHAP values
rf_shap_values = rf_explainer.shap_values(
    X_test_rf_transformed,
    check_additivity=False
)

print("SHAP values calculated successfully!")

print("\nSHAP values type:", type(rf_shap_values))

if isinstance(rf_shap_values, list):

    print("Number of classes:", len(rf_shap_values))

    for i, values in enumerate(rf_shap_values):
        print(
            f"Class {i} shape:",
            values.shape
        )

else:

    print("SHAP values shape:", rf_shap_values.shape)

# Overall SHAP Feature Importance

if isinstance(rf_shap_values, list):

    mean_abs_shap = np.mean(
        [
            np.abs(values).mean(axis=0)
            for values in rf_shap_values
        ],
        axis=0
    )

else:

    mean_abs_shap = np.abs(
        rf_shap_values
    ).mean(axis=(0, 2))

feature_importance = pd.DataFrame({
    "Feature": rf_feature_names,
    "Mean_ABS_SHAP": mean_abs_shap
})

feature_importance = feature_importance.sort_values(
    "Mean_ABS_SHAP",
    ascending=False
)

print("\nRandom Forest SHAP Feature Importance:")
print(feature_importance)

# SHAP Summary Plot

moderate_index = list(rf_classifier.classes_).index("Moderate")
print("\nModerate Risk class index:", moderate_index)

# SHAP Analysis for High Risk

high_index = list(rf_classifier.classes_).index("High")
print("\nHigh Risk class index:", high_index)

# Results Summary

print("\nRisk Level Distribution:")
print(df["Risk_Level"].value_counts())

print("\nCross-Validation Results:")
print(cv_results)

print("\nFinal Test Results:")
print(final_model_results)

print("\nBest Model Based on Macro F1:")
print(best_model_row)

print("\nTop 10 SHAP Features:")
print(feature_importance.head(10))

# Save Results

df.to_csv("cleaned_survey_data.csv", index=False)
cv_results.to_csv("cv_results.csv", index=False)
final_model_results.to_csv("final_model_results.csv", index=False)
feature_importance.to_csv("shap_feature_importance.csv", index=False)
dl_cv_results.to_csv("deep_learning_cv_results.csv", index=False)


joblib.dump(rf_model, "random_forest_model.pkl")