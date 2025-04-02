import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Final_updated_telecom_churn.csv')

# Data Preprocessing
data['numeric_gender'] = data['gender'].map({"M": 1, "F": 2})
data['age'] = data['age'].astype('int64')
data['num_dependents'] = data['num_dependents'].astype('int64')

# Display the initial data overview
st.title("Telecom Churn Analysis")
st.sidebar.header("Navigation")
st.sidebar.markdown("Created by [Marwan Fayad](https://www.linkedin.com/in/marwan-fayad-427314249/)")

sidebar_sel = st.sidebar.radio("Select An Option", ['Introduction', 'EDA', 'Charts'])

if sidebar_sel == 'Introduction':
    st.header("Introduction About The Dataset")
    st.write("This is a web application that allows users to explore the Telecom Churn dataset.")
    st.write("The dataset provides details on customers, their subscription plans, demographics, usage data, and whether they churned (left the service).")
    st.write(data.head())
    st.markdown("### Dataset Summary")
    st.write(data.describe())

    # Filtered Dataset Option inside the Introduction section
    st.markdown("### Filtered Dataset Option")
    show_filtered_data = st.checkbox("Show Filtered Dataset")
    
    if show_filtered_data:
        # Drop the specified columns
        data_filtered = data.drop(columns=['sms_sent', 'calls_made', 'data_used'])
        st.title("Filtered Dataset")
        st.write(data_filtered.head())  # Display the filtered dataset

elif sidebar_sel == 'EDA':
    st.header('Exploratory Data Analysis')
    
    # Distribution of Age
    plt.figure(figsize=(12, 6))
    plt.hist(data['age'], bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    st.pyplot(plt)
    
    # Distribution of Estimated Salary
    plt.figure(figsize=(12, 6))
    plt.hist(data['estimated_salary'], bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Estimated Salary')
    plt.xlabel('Estimated Salary')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    st.pyplot(plt)
    
elif sidebar_sel == 'Charts':
    st.header('Main Objectives Charts')

    # Churned and Non-Churned Customers by Telecom Partner
    churn_by_partner = data.groupby(['telecom_partner', 'churn']).size().unstack().fillna(0)
    churn_by_partner = churn_by_partner.rename(columns={0: 'Not Churn', 1: 'Churn'})

    fig, ax = plt.subplots(figsize=(10, 6))
    churn_by_partner.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'lightgreen'])
    ax.set_title('Churned and Non-Churned Customers by Telecom Partner')
    ax.set_xlabel('Telecom Partner')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
    st.pyplot(fig)
    
    # Churn by Gender
    churn_by_gender = data.groupby('gender')['churn'].value_counts().unstack().fillna(0)
    churn_by_gender = churn_by_gender.rename(columns={0: 'Not Churn', 1: 'Churn'})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    churn_by_gender.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'lightgreen'])
    ax.set_title('Churn Distribution by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    # Churn by Age
    churn_by_age = data.groupby('age')['churn'].value_counts().unstack().fillna(0)
    churn_by_age = churn_by_age.rename(columns={0: 'Not Churn', 1: 'Churn'})
    
    fig, ax = plt.subplots(figsize=(12, 6))
    churn_by_age.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'lightgreen'])
    ax.set_title('Churn Distribution by Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    
    
    # Churned and Non-Churned Customers by City (New Graph)
    city_count = data['city'].value_counts()  # Count customers by city

    fig, ax = plt.subplots(figsize=(12, 6))
    city_count.plot(kind='bar', ax=ax, color='lightcoral')
    ax.set_title('Customer Count by City')
    ax.set_xlabel('City')
    ax.set_ylabel('Customer Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels for better readability
    st.pyplot(fig)

    # Churn by Number of Dependents
    churn_by_dependents = data.groupby('num_dependents')['churn'].value_counts().unstack().fillna(0)
    churn_by_dependents = churn_by_dependents.rename(columns={0: 'Not Churn', 1: 'Churn'})
    
    fig, ax = plt.subplots(figsize=(8, 6))
    churn_by_dependents.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'lightgreen'])
    ax.set_title('Churn Distribution by Number of Dependents')
    ax.set_xlabel('Number of Dependents')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    # Churn Prediction Visualization
    st.header('Churn Prediction Model')
    
    # Assuming you have the 'predicted_churn' column from your decision tree model
    # Display the first few rows of predicted churn values
    st.write(data[['customer_id', 'churn', 'predicted_churn']].head())
    
    # Plot actual vs predicted churn
    plt.figure(figsize=(8, 6))
    plt.scatter(data['churn'], data['predicted_churn'], color='blue', alpha=0.6)
    plt.title("Actual vs Predicted Churn")
    plt.xlabel("Actual Churn")
    plt.ylabel("Predicted Churn")
    plt.grid(True)
    st.pyplot(plt)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(data['churn'], data['predicted_churn'])
    
    plt.figure(figsize=(8, 6))
    plt.matshow(cm, cmap='Blues', fignum=1)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Not Churn', 'Churn'])
    plt.yticks([0, 1], ['Not Churn', 'Churn'])
    st.pyplot(plt)

    # Classification Report
    from sklearn.metrics import classification_report
    report = classification_report(data['churn'], data['predicted_churn'])
    st.text(report)
