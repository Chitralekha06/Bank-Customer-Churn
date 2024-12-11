Customer Churn Prediction Project

OVERVIEW:

This project focuses on predicting customer churn for a European multinational bank operating in France, Germany, and Spain. The bank faces high churn rates, leading to revenue losses and increased customer acquisition costs. The goal is to identify factors contributing to churn and develop predictive models to mitigate this issue by retaining customers.

PROBLEM STATEMENT:

Customer churn is a critical problem for the bank, driven by factors such as competition, dissatisfaction, changing needs, and poor service. Understanding why customers leave is essential for taking proactive measures, such as offering personalized services and improving engagement. By reducing churn, the bank aims to enhance customer retention and strengthen its market position.

APPROACH:

The project employs a systematic approach divided into the following stages:

1. Data Exploration and Understanding

Initial Analysis: Understanding the dataset’s structure, identifying key variables, and recognizing patterns or anomalies.

Exploratory Data Analysis (EDA): Visualizing relationships and distributions to gain insights into customer behavior and churn trends.

2. Data Preprocessing

Data Cleaning: Removing or imputing missing values, handling outliers, and ensuring consistency across data entries.

Feature Engineering: Creating new features and transforming existing ones to improve predictive power.

Data Transformation: Encoding categorical variables, normalizing numerical variables, and splitting data into training and testing sets.

Addressing Class Imbalance: Employing SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset, ensuring models can accurately predict churn without bias.

3. Model Development and Evaluation

Model Selection: Exploring machine learning models such as Logistic Regression, Random Forest, and Gradient Boosting techniques (e.g., XGBoost).

Assumption Validation: For Logistic Regression, ensuring binary targets, low multicollinearity, and linear relationships between predictors and log-odds.

Model Tuning: Optimizing hyperparameters using grid search or random search techniques to enhance model performance.

Performance Metrics: Evaluating models using metrics such as Accuracy, Precision, Recall, F1-score, and AUC-ROC to determine the best model.

4. Best Model Selection and Deployment

Comparison: Comparing model results before and after applying techniques like SMOTE.

Best Model Evaluation: Highlighting the best-performing model’s accuracy and implications for business decisions.

NOTEBOOKS:

The project is organized into the following Jupyter notebooks:

Preprocessing Notebook (DSE-FT-C-MAY24-G5-FinalReport_Preprocessing.ipynb)

Summarizes the problem statement, data, and findings.

Prepares the dataset for modeling by handling missing values, encoding categorical variables, and addressing class imbalance.

Model Development Notebook (DSE-FT-C-MAY24-G5-FinalReport_models.ipynb)

Installs necessary packages like xgboost and scikit-learn.

Develops machine learning models and evaluates their performance before and after applying SMOTE.

Best Model Notebook (DSE-FT-C-May24-G5-BestModel.ipynb)

Focuses on the assumptions and requirements for logistic regression.

Evaluates the best-performing model for the given problem.

KEY FINDINGS AND RESULTS:

Insights from EDA: Discovered key factors influencing customer churn, such as account balance, tenure, and customer satisfaction scores.

Model Performance: XGBoost emerged as the best-performing model with a high AUC-ROC score, effectively predicting high-risk customers.

Business Implications: The bank can target high-risk customers with personalized interventions to reduce churn and improve customer retention rates.

DEPENDENCIES:

Python 3.x

Jupyter Notebook

scikit-learn

xgboost

pandas

numpy

matplotlib

seaborn

HOW TO RUN:

Clone the repository.

Install the required dependencies listed in requirements.txt.

Open the notebooks in Jupyter Notebook or Jupyter Lab.

Follow the steps in each notebook to preprocess the data, build models, and evaluate results.

FUTURE ENHANCEMENTS:

Incorporating additional data sources, such as customer feedback or transaction history, to enhance model accuracy.

Experimenting with deep learning models for improved predictions.

Developing a user-friendly dashboard for real-time churn prediction.
