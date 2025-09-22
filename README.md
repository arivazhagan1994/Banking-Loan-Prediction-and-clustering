# Banking-Loan-Prediction-and-clustering
Loan Prediction Model using Machine Learning (XGBoost, Random Forest, Logistic Regression) with preprocessing, hyperparameter tuning)

# Loan Risk Prediction Project README

## Project Overview
This project aims to build a machine learning model to predict loan risk based on various applicant features. By analyzing historical loan data, the model will classify loan applications as either 'High Risk' or 'Low Risk', helping financial institutions make informed lending decisions and mitigate potential losses.

## Problem Statement
The primary problem addressed is the need for an automated and accurate system to assess the risk associated with loan applications. Manual risk assessment can be time-consuming, inconsistent, and prone to human error. A reliable predictive model is crucial for efficient and effective risk management in the lending process.

## Objectives
The main objectives of this project are:
- To explore and understand the loan application dataset.
- To preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
- To perform exploratory data analysis to identify patterns, relationships, and insights within the data.
- To engineer relevant features that can improve model performance.
- To build and evaluate various machine learning models for loan risk prediction.
- To select the best-performing model based on appropriate evaluation metrics.
- To tune and optimize the selected model's hyperparameters for better performance.
- To understand the model's predictions through explainability techniques.
- To validate the model's robustness and generalization ability.
- To outline a deployment strategy for the final model.
- To identify the limitations of the current model and suggest future enhancements.

## Dataset Description
The dataset contains information about loan applicants and includes features such as Income, Age, Experience, Marital Status, House Ownership, Car Ownership, Profession, City, State, Current Job Years, Current House Years, and the target variable, Risk Flag (0 for Low Risk, 1 for High Risk). The dataset has 252000 rows and 13 columns.

## Data Preprocessing
Data preprocessing was performed to prepare the raw data for machine learning models. This involved:

### Handling Missing Values
No missing values were found in this dataset.

### Encoding Categorical Variables
- **Label Encoding:** Applied to 'Married/Single' and 'Age_Group'.
- **One-Hot Encoding:** Applied to 'House_Ownership', 'Car_Ownership', 'Zone', and 'Profession_Category'.

### Feature Scaling & Normalization
Numerical features ('Income', 'Experience', 'CURRENT_JOB_YRS', 'CURRENT_HOUSE_YRS') were scaled using `StandardScaler`.

## Exploratory Data Analysis (EDA)
EDA was performed to gain insights into the dataset's characteristics and relationships between variables. Visualizations were used to analyze distributions, correlations, and the relationship between features and the target variable.

## Feature Engineering
New features were created:
- **Zone:** Created from the 'STATE' column.
- **Profession_Category:** Created by grouping similar professions.
- **Age_Group:** Created by grouping ages into bins.

The original 'STATE', 'Profession', 'CITY', and 'Age' columns were dropped. The 'Id' column was also dropped.

## Model Selection and Evaluation
Several machine learning models were considered and evaluated based on Accuracy, Precision, Recall, and F1-Score.

- **Logistic Regression:** Achieved an F1 Score of 0.23.
- **Random Forest:** Achieved an F1 Score of 0.43.
- **XGBoost:** Achieved an F1 Score of 0.57.

## Hyperparameter Tuning & Optimization
`GridSearchCV` with Stratified K-Fold cross-validation was used to tune the hyperparameters of the XGBoost model. The best parameters found were {'learning_rate': 0.2, 'max_depth': 10, 'n_estimators': 200, 'scale_pos_weight': 3}, resulting in a cross-validation F1 score of 0.64 and a test set F1 score of 0.64.

## Model Validation
The ROC Curve and AUC were used to validate the best model's performance. The AUC score was 0.89.

## Final Model Selection
Based on the evaluation metrics, the tuned XGBoost model was selected as the final model.

## Deployment Strategy
The trained model (`loan_risk_model.pkl`) and the scaler (`scaler.pkl`) were saved using `joblib` for future deployment. The strategy includes using the saved scaler and applying the same encoding steps to new data before making predictions. The model's output (0 or 1) will be translated to 'Low Risk' or 'High Risk', and probability scores can also be used.

## Conclusion
This project successfully developed and evaluated an XGBoost model for loan risk prediction, demonstrating promising performance. The outlined deployment strategy provides a path for practical application.

## How to Run the Code
1. Ensure you have the necessary libraries installed (`dtale`, `missingno`, `xgboost`, `joblib`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`). You can install them using pip: `pip install dtale missingno xgboost joblib pandas numpy matplotlib seaborn sklearn`
2. Load the dataset named "Loan Prediction.csv".
3. Run the code cells sequentially to perform data loading, exploration, preprocessing, model training, evaluation, tuning, and saving the model.

## Future Enhancements
- Explore other advanced machine learning models.
- Implement more sophisticated feature engineering techniques.
- Investigate imbalanced data handling techniques further.
- Develop a user interface for making predictions on new data.
- Incorporate model explainability techniques (e.g., SHAP values) for better understanding of predictions.


# Bank Transaction Analysis and Customer Segmentation

## Project Overview

This project analyzes bank transaction data to understand customer behavior and identify potential customer segments using clustering techniques. The goal is to provide insights that can inform targeted business strategies.

## Dataset

The analysis is based on the `bank_transactions.csv` dataset, which contains information about bank transactions, including customer details, transaction amounts, and dates.

## Analysis Steps

The project involved the following key steps:

1.  **Data Loading and Understanding:** Loading the dataset and performing initial exploration to understand its structure and identify missing values.
2.  **Data Preprocessing:** Handling missing values, converting data types, calculating customer age from date of birth, and removing outliers.
3.  **Feature Engineering:** Creating new features such as 'CustomerAge' and 'Zone' based on customer location.
4.  **Categorical Data Encoding:** Converting categorical features ('CustGender' and 'Zone') into numerical representations using Label Encoding and One-Hot Encoding.
5.  **Exploratory Data Analysis (EDA):** Analyzing the distribution of numerical features, checking for skewness, and examining correlations between features.
6.  **Dimensionality Reduction:** Applying Principal Component Analysis (PCA) to reduce the dimensionality of the data for visualization.
7.  **Clustering:** Using K-Means clustering to group customers into segments based on their characteristics. The optimal number of clusters was determined using the Elbow Method and Silhouette Score.
8.  **Visualization:** Visualizing the clusters in a 2D space using the results of PCA.

## Key Findings (Based on the provided analysis)

*   Missing values in `CustGender`, `CustLocation`, and `CustAccountBalance` were handled.
*   Invalid and underage customer entries based on `CustomerDOB` were removed.
*   Outliers in numerical features were addressed using the IQR method.
*   The analysis identified 2 customer clusters using K-Means, although the clusters were not distinctly separated in the 2D PCA visualization.

## Potential Next Steps

*   Explore other clustering algorithms (e.g., DBSCAN, Hierarchical Clustering) to potentially identify different or more clearly defined clusters.
*   Perform a detailed analysis of the characteristics of the identified clusters to understand the profiles of each customer segment.
*   Consider using t-SNE for visualizing the clusters, as it can sometimes reveal non-linear relationships better than PCA.
*   Translate the clustering insights into actionable business strategies.

## How to Run the Code

1.  Ensure you have the `bank_transactions.csv` file available and update the `file_path` variable in the notebook to point to the correct location.
2.  Run the cells sequentially in the provided Jupyter/Colab notebook.

## Dependencies

The following libraries are required to run the notebook:

*   pandas
*   numpy
*   datetime
*   sklearn
*   matplotlib
*   seaborn
*   rapidfuzz
*   tqdm
*   scipy
