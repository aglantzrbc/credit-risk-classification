# credit-risk-classification

## Overview of the Analysis
* The purpose of the analysis was to build a model that can identify the creditworthiness of borrowers based on historical lending activity data for a peer-to-peer lending services company. Supervised machine learning techniques were used to accomplish this goal.
* The financial data included these fields: `loan_size`, `interest_rate`,	`borrower_income`, `debt_to_income`,	`num_of_accounts`,	`derogatory_marks`, `total_debt`, and `loan_status`. It's assumed that the first seven datapoints are the basis for generating the `loan_score` value, which determines the overall disposition of the particular loan. For this analysis, the first seven fields were therefore collectively employed as an independent variable to predict the eighth field, `loan_status`, the dependent variable.
* The analysis involves a binary classification. The dependent variable, `loan_status`, can only take one of two discrete status values: `0` for `Healthy Loan`, presumably a loan that counts in the applicant's favor for future lending, and '1' for `High-Risk Loan`, which is probably a flag for special scrutiny by the lender.
* The analysis proceeded as follows:
  -  The lending history data was read into a Python Pandas DataFrame
  -  The data was split into the `y` (dependent) variable, or _label_ (i.e., `loan_status`) values and the X (independent) variable, or _feature_, values.
  -  The volumes by `y` were verified
  -  The data was split into training and testing subsets using the _train_test_split_ function
  -  A logistic regression model was instantiated using the _LogisticRegression_ classifier and fitted using the training data
  -  The model was then used to make predictions feom the testing data
  -  The model's performance was evaluated by calculating its accuracy score, generating a confusion matrix, and creating a classification report
  -  In a subsequent round, _RandomOverSampler_ resamples the data to make the quantities artificially equal for each value of `y`
  -  A new logistic regression model was instantiated using the _LogisticRegression_ classifier and fitted using the resampled training data
  -  The model was then used to make new predictions feom the testing data
  -  The revised model's performance was evaluated by calculating its accuracy score, generating a confusion matrix, and creating a classification report
* The analysis used the following two methods:
  -  In the first iteration, the data was used as-is, even though there was a large imbalance in volume between values of the dependent variable. _train_test_split_ was used to divide the data into training and testing batches, the _LogisticRegression_ module created a predictive model, the training data was fitted to it, and predictions were made on the testing data.
  -  In the second round, the training data was artificially resampled using the _RandomOverSampler_ function, so that both possible values of `y` have the same volume. As before, the _LogisticRegression_ module created a predictive model, the resampled training data was fitted to it, and new predictions were made on the testing data.

## Results:
* **Key performance indicators definitions:**
  - **Accuracy:** This is the proportion of the total number of predictions that were correct. It is calculated as (True Positives + True Negatives) / Total Observations. If the dataset is imbalanced, it's better to use a **balanced accuracy** score, which is calculated as the arithmetic mean of sensitivity (a.k.a., recall - the true positive rate) and specificity (true negative rate), effectively taking both classes into account in a balanced manner. Balanced accuracy compensates for the bias in favor of the majority class by giving equal weight to each class's sensitivity.
  - **Precision:** This is the ratio of correctly predicted positive observations to the total predicted positives (True Positives / (True Positives + False Positives)). High precision indicates that false positive error is low.
  - **Recall:** Also known as sensitivity, this is the ratio of correctly predicted positive observations to the all observations in actual class. The formula is (True Positives / (True Positives + False Negatives)).

* **Machine Learning Model 1 - data used as-is, with imbalanced volumes for the two values of the `y` variable:**
  - **Accuracy**: The accuracy score is a nearly perfect 99%. The balanced accuracy score is a little lower, reflecting the dataset's bias toward healthy loans, though still high at 94%. Predictions were almost always correct, with only 147 incorrect out of 19,384 cases.
  - **Precision:** A commanding 100% of loans the model predicted as healthy was actually healthy, but a less impressive 87% of loans the model predicted as high-risk were actually high-risk. The model is considerably better at predicting healthy than high risk loans.
  - **Recall:** 100% of the healthy loans in the dataset were identified correctly as healthy, but a lower proportion (89%) of the dataset's high-risk loans were identified as such. Presumably, 11% of the dataset's high-risk loans were incorrectly classified as healthy.

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
