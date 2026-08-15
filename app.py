import streamlit as st
import pandas as pd
import joblib

# Page configuration

st.set_page_config(
    page_title="ScrollSense",
    page_icon="📱",
    layout="wide"
    )

# Load model and results

model = joblib.load("random_forest_model.pkl")

final_model_results = pd.read_csv("final_model_results.csv")

cv_results = pd.read_csv("cv_results.csv")

shap_features = pd.read_csv("shap_feature_importance.csv")

cleaned_data = pd.read_csv("cleaned_survey_data.csv")

# Title

st.title("ScrollSense")

st.subheader("Short-Video Usage Risk Assessment")

st.write(
    """
    ScrollSense is a machine learning project that estimates
    short-video usage risk levels based on survey responses.
    
    The model classifies participants into three categories:
    **Low, Moderate, or High Risk.**
    """
)

st.warning(
    "This tool is intended for educational and research purposes. "
    "It is not a clinical or psychological diagnosis."
)

# Sidebar

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Risk Assessment",
        "Survey Insights",
        "Model Performance",
        "About"
    ]
)

# Risk Assessment

if page == "Risk Assessment":

    st.header("Risk Assessment")

    st.write(
        "Enter the following information to estimate the short-video "
        "usage risk level."
    )

    col1, col2= st.columns(2)

    with col1:

        age= st.selectbox(
            "Age",
            sorted(cleaned_data["Age"].dropna().unique())
        )

        gender= st.selectbox(
            "Gender",
            sorted(cleaned_data["Gender"].dropna().unique())
        )

        occupation= st.selectbox(
            "Occupation",
            sorted(cleaned_data["Occupation"].dropna().unique())
        )

        platforms= st.selectbox(
            "Platforms Used",
            sorted(cleaned_data["Platforms_Used"].dropna().unique())
        )

        daily_usage= st.selectbox(
            "Daily Short-Video Usage",
            sorted(cleaned_data["Daily_Usage"].dropna().unique())
        )

    with col2:

        daily_app_opens= st.selectbox(
            "Daily App Opens",
            sorted(cleaned_data["Daily_App_Opens"].dropna().unique())
        )

        typical_usage_time= st.selectbox(
            "Typical Usage Time",
            sorted(cleaned_data["Typical_Usage_Time"].dropna().unique())
        )

        main_usage_reason= st.selectbox(
            "Main Usage Reason",
            sorted(cleaned_data["Main_Usage_Reason"].dropna().unique())
        )

        session_duration= st.selectbox(
            "Average Session Duration",
            sorted(
                cleaned_data[
                    "Average_Session_Duration"
                ].dropna().unique()
            )
        )

    if st.button(
        "Assess Risk",
        use_container_width=True):

        input_data = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Occupation": [occupation],
            "Platforms_Used": [platforms],
            "Daily_Usage": [daily_usage],
            "Daily_App_Opens": [daily_app_opens],
            "Typical_Usage_Time": [typical_usage_time],
            "Main_Usage_Reason": [main_usage_reason],
            "Average_Session_Duration": [session_duration]
        })

        prediction= model.predict(input_data)[0]

        st.divider()

        st.subheader("Estimated Risk Level")

        if prediction == "High":

            st.error("🔴 High Risk")

        elif prediction == "Moderate":

            st.warning("🟠 Moderate Risk")

        else:

            st.success("🟢 Low Risk")

        st.info(
            "This prediction is based on patterns learned from "
            "the survey dataset."
        )

# Survey Insights

elif page == "Survey Insights":

    st.header("Survey Insights")

    st.write("Overview of the survey data used in the project.")

    col1, col2, col3= st.columns(3)

    total_participants= len(cleaned_data)

    low_count = (
        cleaned_data["Risk_Level"]
        .value_counts()
        .get("Low", 0))

    moderate_count = (
        cleaned_data["Risk_Level"]
        .value_counts()
        .get("Moderate", 0)
    )

    high_count = (
        cleaned_data["Risk_Level"]
        .value_counts()
        .get("High", 0)
    )

    with col1:

        st.metric(
            "Participants",
            total_participants
        )

    with col2:

        st.metric(
            "Moderate Risk",
            moderate_count
        )

    with col3:

        st.metric(
            "High Risk",
            high_count
        )

    st.subheader(
        "Risk Level Distribution"
    )

    risk_counts = (
        cleaned_data["Risk_Level"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High"]
        )
        .fillna(0)
    )

    st.bar_chart(risk_counts)

    st.subheader("Average Risk Score by Daily Usage")

    daily_usage_results = (
        cleaned_data
        .groupby("Daily_Usage")["Risk_Score"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(daily_usage_results)

    st.subheader("Average Risk Score by Session Duration")

    session_results = (
        cleaned_data
        .groupby(
            "Average_Session_Duration"
        )["Risk_Score"]
        .mean()
        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(session_results)

# Model Performance

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "Comparison of the machine learning models evaluated "
        "during the project."
    )

    st.subheader("Final Test Results")

    st.dataframe(
        final_model_results,
        use_container_width=True
    )

    st.subheader("Macro F1 Comparison")

    st.bar_chart(
        final_model_results.set_index(
            "Model"
        )["Macro F1"]
    )

    st.subheader("5-Fold Cross-Validation Results")

    st.dataframe(
        cv_results,
        use_container_width=True
    )

    st.subheader("Top SHAP Features")

    st.dataframe(
        shap_features.head(10),
        use_container_width=True
    )

# About

elif page == "About":

    st.header("ℹ️ About ScrollSense")

    st.write(
        """
        ScrollSense was developed as a machine learning project
        exploring whether demographic and short-video usage
        characteristics can be used to estimate behavioral
        usage-risk levels.
        """
    )

    st.subheader("Machine Learning Models")

    st.write(
        """
        The project evaluated:

        • Logistic Regression  
        • Random Forest  
        • XGBoost  
        • Deep Learning  

        Five-fold stratified cross-validation was used to evaluate
        model performance, followed by XGBoost hyperparameter tuning.
        """
    )

    st.subheader("Explainability")

    st.write(
        """
        SHAP was used to examine which features contributed most
        to the Random Forest model's predictions.
        """
    )

    st.subheader("Important Limitation")

    st.write(
        """
        The dataset contains 100 survey responses, with fewer
        participants in the High-risk category. Therefore,
        predictions—especially for the High-risk class—should be
        interpreted cautiously.
        """
    )