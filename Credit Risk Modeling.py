# Created by ivywang at 2024-02-28
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    print(cr_loan.columns[cr_loan.isnull().any()])

    # Print the top five rows with nulls for employment length
    print(cr_loan[cr_loan['person_emp_length'].isnull()].head())

    # Impute the null values with the median value for all employment lengths
    cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

    # Create a histogram of employment length
    n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
    plt.xlabel("Person Employment Length")
    plt.show()

    ## drop NaN
    # Print the number of nulls
    print(cr_loan['loan_int_rate'].isnull().sum())

    # Store the array on indices
    indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index

    # Save the new data without missing data
    cr_loan_clean = cr_loan.drop(indices)


def main():
    # credit loan data
    cr_loan = data_extraction('cr_loan.csv')

    ## Credit Risk Modeling ##
    # cross_table(cr_loan)
    # outlier_scatter(cr_loan)
    missing_data(cr_loan)



'''Main Function'''
if __name__ == '__main__':
    main()
