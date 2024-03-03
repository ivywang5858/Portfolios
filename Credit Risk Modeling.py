# Created by ivywang at 2024-02-28
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_fscore_support

def data_extraction(file):
    data = pd.read_csv(file)
    return data

# Exploring and Preparing Loan Data
def cross_table(cr_loan):
    # Create a cross table of the loan data
    print(pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins=True))
    print(pd.crosstab(cr_loan['loan_status'], cr_loan['person_home_ownership'],
                       values=cr_loan['person_emp_length'], aggfunc='max'))

# Outlier
def outlier_scatter(cr_loan):
    # Create the scatter plot for age and amount
    plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)
    plt.xlabel("Person Age")
    plt.ylabel("Loan Amount")
    plt.show()

# Replacing missing credit data
def missing_data(cr_loan):
    # Print a null value column array
    # print(cr_loan.columns[cr_loan.isnull().any()])

    # Print the top five rows with nulls for employment length
    # print(cr_loan[cr_loan['person_emp_length'].isnull()].head())

    # Impute the null values with the median value for all employment lengths
    cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

    # Create a histogram of employment length
    n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
    plt.xlabel("Person Employment Length")
    # plt.show()

    ## drop NaN
    # Print the number of nulls
    # print(cr_loan['loan_int_rate'].isnull().sum())

    # Store the array on indices
    indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index

    # Save the new data without missing data
    cr_loan_clean = cr_loan.drop(indices)
    return cr_loan_clean

def logistic_regression(cr_loan_clean):
    # Logistic regression basics
    # Create the X and y data sets
    X = cr_loan_clean[['loan_int_rate']]
    y = cr_loan_clean[['loan_status']]

    # Create and fit a logistic regression model
    clf_logistic_single = LogisticRegression()
    clf_logistic_single.fit(X, np.ravel(y))

    # Print the parameters of the model
    print(clf_logistic_single.get_params())
    # Print the intercept of the model
    print(clf_logistic_single.intercept_)

    # Multivariate logistic regression
    # Create X data for the model
    X_multi = cr_loan_clean[['loan_int_rate', 'person_emp_length']]
    # Create a set of y data for training
    y = cr_loan_clean[['loan_status']]
    # Create and train a new logistic regression
    clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))

    # Print the intercept of the model
    print(clf_logistic_multi.intercept_)
    print(clf_logistic_multi.coef_)

def test_and_train(cr_loan_clean):
    # Creating training and test sets
    # Create the X and y data sets
    X = cr_loan_clean[['loan_int_rate', 'person_emp_length', 'person_income']]
    y = cr_loan_clean[['loan_status']]
    # Use test_train_split to create the training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

    # Create and fit the logistic regression model
    clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))
    # Print the models coefficients
    print(clf_logistic.coef_)

def one_hot_encoding(cr_loan_clean):
    # prepare the non-numeric columns
    # Create two data sets for numeric and non-numeric data
    cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
    cred_str = cr_loan_clean.select_dtypes(include=['object'])

    # One-hot encode the non-numeric columns
    cred_str_onehot = pd.get_dummies(cred_str)

    # Union the one-hot encoded columns to the numeric ones
    cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

    # Print the columns in the new data set
    print(cr_loan_prep.columns)
    return cr_loan_prep

def pred_prob_of_def(cr_loan_clean):
    # Create the X and y data sets
    X = cr_loan_clean[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'person_home_ownership_MORTGAGE',
       'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT', 'loan_intent_DEBTCONSOLIDATION', 'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT',
       'loan_intent_MEDICAL', 'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'loan_grade_A', 'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 'loan_grade_F', 'loan_grade_G',
       'cb_person_default_on_file_N', 'cb_person_default_on_file_Y']]
    y = cr_loan_clean[['loan_status']]
    # Use test_train_split to create the training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)
    # Train the logistic regression model on the training data
    clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

    # Create predictions of probability for loan status using test data
    preds = clf_logistic.predict_proba(X_test)
    # Create dataframes of first five predictions, and first five true labels
    preds_df = pd.DataFrame(preds[:, 1][0:5], columns=['prob_default'])
    true_df = y_test.head()

    # Concatenate and print the two data frames for comparison
    print(pd.concat([true_df.reset_index(drop=True), preds_df], axis=1))

def default_classification_report(cr_loan_clean):
    X = cr_loan_clean[['loan_int_rate', 'person_emp_length', 'person_income']]
    y = cr_loan_clean[['loan_status']]
    # Use test_train_split to create the training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)
    # Train the logistic regression model on the training data
    clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))
    # Create predictions of probability for loan status using test data
    preds = clf_logistic.predict_proba(X_test)

    # Create a dataframe for the probabilities of default
    preds_df = pd.DataFrame(preds[:, 1], columns=['prob_default'])
    # Reassign loan status based on the threshold
    preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

    # Print the row counts for each loan status
    print(preds_df['loan_status'].value_counts())
    # Print the classification report
    target_names = ['Non-Default', 'Default']
    print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))

    # Visually scoring credit models
    # Create predictions and store them in a variable
    preds = clf_logistic.predict_proba(X_test)

    # Print the accuracy score the model
    print(clf_logistic.score(X_test, y_test))

    # Plot the ROC curve of the probabilities of default
    prob_default = preds[:, 1]
    fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
    plt.plot(fallout, sensitivity, color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.show()

    # Compute the AUC and store it in a variable
    auc = roc_auc_score(y_test, prob_default)
def default_recall_rate(cr_loan_clean):
    X = cr_loan_clean[['loan_int_rate', 'person_emp_length', 'person_income']]
    y = cr_loan_clean[['loan_status']]
    # Use test_train_split to create the training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)
    # Train the logistic regression model on the training data
    clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))
    # Create predictions of probability for loan status using test data
    preds = clf_logistic.predict_proba(X_test)

    # Create a dataframe for the probabilities of default
    preds_df = pd.DataFrame(preds[:, 1], columns=['prob_default'])
    # Reassign the values of loan status based on the new threshold
    preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

    # Store the number of loan defaults from the prediction data
    num_defaults = preds_df['loan_status'].value_counts()[1]

    # Store the default recall from the classification report
    default_recall = precision_recall_fscore_support(y_test, preds_df['loan_status'])[1][1]

    avg_loan_amnt = 9583.600936895346
    # Calculate the estimated impact of the new default recall rate
    print(avg_loan_amnt * num_defaults * (1 - default_recall))
def main():
    # credit loan data
    cr_loan = data_extraction('cr_loan.csv')

    ## Credit Risk Modeling ##
    # cross_table(cr_loan)
    # outlier_scatter(cr_loan)
    # missing_data(cr_loan)

    cr_loan_clean = data_extraction('cr_loan_nout_nmiss.csv')
    # logistic_regression(cr_loan_clean)
    # test_and_train(cr_loan_clean)
    # cr_loan_prep = one_hot_encoding(cr_loan_clean)
    # pred_prob_of_def(cr_loan_prep)
    # default_classification_report(cr_loan_clean)
    default_recall_rate(cr_loan_clean)

'''Main Function'''
if __name__ == '__main__':
    main()
