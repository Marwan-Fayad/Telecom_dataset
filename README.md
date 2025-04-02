# Telecom_dataset
Telecom Churn Dataset
This dataset contains information about telecom customers, their demographic data, service details, and whether they churned (left the service). It is primarily used for analyzing customer churn patterns, predicting churn, and conducting exploratory data analysis.

Dataset Features
customer_id: Unique identifier for each customer.

gender: Gender of the customer (M: Male, F: Female).

age: Age of the customer (in years).

state: State in which the customer resides.

city: City where the customer lives.

pincode: Postal code of the customer's address.

date_of_registration: Date when the customer registered for the service.

num_dependents: Number of dependents the customer has.

estimated_salary: Estimated annual salary of the customer.

churn: Target variable, indicates whether the customer has churned (1) or not (0).

sms_sent: Total number of SMS sent by the customer.

calls_made: Total number of calls made by the customer.

data_used: Total amount of data used by the customer (in GB).

predicted_calls_made: Predicted value for the total number of calls made by the customer (used to replace negative values).

predicted_sms_sent: Predicted value for the total number of SMS sent by the customer (used to replace negative values).

predicted_data_used: Predicted value for the total amount of data used by the customer (used to replace negative values).

numeric_gender: Numeric encoding of the gender (1 for Male, 2 for Female).

age_salary_interaction: Interaction feature between age and estimated salary (age * salary).

age_dependents_interaction: Interaction feature between age and number of dependents (age * dependents).

calls_sms_interaction: Interaction feature between calls made and SMS sent.

log_estimated_salary: Log-transformed value of the estimated salary.

age^2: Squared value of age to capture non-linear relationships.

age estimated_salary: Combined feature of age and estimated salary (age * estimated_salary).

estimated_salary^2: Squared value of the estimated salary to capture non-linear relationships.

Data Cleaning and Preprocessing
The dataset originally contained negative values in the sms_sent, calls_made, and data_used columns. These negative values were identified as erroneous data. To handle this, the negative values were replaced with NaN values.

To impute these missing values, we applied machine learning models, specifically regression models (e.g., Decision Trees, Random Forests, etc.), to predict the missing values based on other available features. The predicted values from these models were then used to replace the NaN values in the dataset.

Dataset Overview
This dataset was collected for analyzing customer behavior, with a focus on predicting customer churn. The goal is to understand patterns that can indicate if a customer is likely to churn, so telecom companies can take proactive measures to retain valuable customers.

Usage
This dataset can be used for various machine learning tasks, including:

Churn Prediction: Predicting whether a customer will churn or stay.

Exploratory Data Analysis (EDA): Analyzing customer demographics, usage patterns, and service subscriptions.

Feature Engineering: Creating new features that may help improve the model's performance.

Dataset Insights
Some possible insights that can be derived from the dataset include:

Age distribution of churned vs. non-churned customers.

The effect of salary on customer churn.

Identifying the most common reasons for churn across different regions or service plans.

Important Notes
Negative values in the sms_sent, calls_made, and data_used columns were replaced with NaN values.

Machine learning models were used to predict the missing values in these columns, and the predicted values were used to impute the missing data.

The predicted_columns (such as predicted_calls_made, predicted_sms_sent, and predicted_data_used) were used to compute the final predicted_churn.

