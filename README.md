# credit-risk-classification

## Overview of the Analysis
* The purpose of the analysis was to build a model that can identify the creditworthiness of borrowers based on historical lending activity data for a peer-to-peer lending services company. Supervised machine learning techniques were used to accomplish this goal.
* The financial data included these fields: `loan_size`, `interest_rate`,	`borrower_income`, `debt_to_income`,	`num_of_accounts`,	`derogatory_marks`, `total_debt`, `loan_status`.

* ## Overview of the Analysis

* The purpose of this analysis is to build a model that can identify the creditworthiness of borrowers.
* A dataset of historical lending activity from a peer-to-peer lending services company was used. 
* Dependant variable (y value) in this analysis was the "loan status" indicating if a loan is healthy or at risk. 
* Independent Variables (x values) were loan size, interest rate, borrower income, debt to income ratio, number of accounts and derogatory marks.
* In this analysis, we first split our data to traning and test sets. Then, define our dependent and independent variables. Next, we create logistic regression model and fit our original data to this model. Trained model is used to make predictions. Lastly, we evaluate the model`s performance. 
* Two diffeent Logistic Regression models were created by using the original data set and randomy over resampled data set (to get rid of the imbalances). In the end, their results -which was gathered with scikit-learn library- were compared.

The dataset (77,536 data points) was split into training and testing sets. The training set was used to build an initial logistic regression model (Logistic Regression Model 1) using the `LogisticRegression` module from <a href=https://scikit-learn.org/stable/index.html>scikit-learn</a>. Logistic Regression Model 1 was then applied to the testing dataset. The purpose of the model was to determine whether a loan to the borrower in the testing set would be low- or high-risk and results are summarized below.

This intial model was drawing from a dataset that had 75,036 low-risk loan data points and 2,500 high-risk data points. To resample the training data and ensure that the logistic regression model had an equal number of data points to draw from, the training set data was resampled with the `RandomOverSampler` module from <a href=https://imbalanced-learn.org/dev/index.html>imbalanced-learn</a>. This generated 56,277 data points for both low-risk (0) and high-risk (1) loans, based on the original dataset.

The resampled data was used to build a new logistic regression model (Logistic Regression Model 2). The purpose of Logistic Regression Model 2 was to determine whether a loan to the borrower in the testing set would be low- or high-risk. The results are summarized below.

## Results

<strong>Logistic Regression Model 1:</strong>

* Precision: 93% (an average--in predicting low-risk loans, the model was 100% precise, though the model was only 87% precise in predicting high-risk loans)
* Accuracy: 94% 
* Recall: 95% (an average--the model had 100% recall in predicting low-risk loans, but 89% recall in predicting high-risk loans)

<strong>Logistic Regression Model 2:</strong>

* Precision: 93% (an average--in predicting low-risk loans, the model was 100% precise, though the model was only 87% precise in predicting high-risk loans)
* Accuracy: 100% 
* Recall: 100%

## Summary

Logistic Regression Model 2 is less likely to predict false negative results. However, based on the confusion matrices for each model, Logistic Regression Model 2 predicted slightly more false positives (low-risk when the actual was high-risk). 

If the goal of the model is to determine the likelihood of high-risk loans, neither model scores above 90% precision. Logistic Regression Model 2 had fewer false predictions of the testing data overall and would be the best model to use based on the high accuracy and recall of this model.
