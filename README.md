
# Smart Lender - Applicant Credibility Prediction for Loan Approval

Loans are the core commercial enterprise of banks. The primary earnings comes without delay from the loan’s interest. The loan corporations furnish a mortgage after an extensive method of verification and validation. 

However, they nevertheless don’t have warranty if the applicant is capable of pay off the mortgage with no problems.
In this academic, we’ll construct a predictive model to are expecting if an applicant is able to pay off the lending company or no longer.


## Purpose

The prediction of credit defaulters is one of the difficult tasks for any bank. But by forecasting the loan defaulters, the banks definitely may reduce their loss by reducing their non-profit assets, so that recovery of approved loans can take place without any loss and it can play as the contributing parameter of the bank statement. This makes the study of this loan approval prediction important. Machine Learning techniques are very crucial and useful in the prediction of these types of data.
## Understanding the problem statement

Dream Housing Finance corporation offers in all sorts of domestic loans. They have a presence across all urban, semi-urban and rural regions. The consumer first applies for a home loan and after that, the enterprise validates the client eligibility for the loan.

The company wants to automate the loan eligibility system (real-time) based on client element furnished at the same time as filling out on line software paperwork. These details are Gender, Marital Status, Education, number of Dependents, Income, Loan Amount, Credit History, and others

To automate this process, they have furnished a dataset to pick out the patron segments which can be eligible for mortgage quantities a good way to particularly target those clients.

You can discover the whole details about the hassle announcement right here and also download the training and check records.

As noted above this is a Binary Classification trouble wherein we need to predict our Target label that's “Loan Status”.

Loan fame may have  values: Yes or NO.  
Yes: if the loan is approved  
NO: if the loan is not approved

So using the training dataset we will teach our version and try to are expecting our target column that is “Loan Status” on the take a look at dataset.

## Project Flow

Install Required Libraries.

Data Collection:  
·      Collect the dataset or Create the dataset

Data Preprocessing:  
·      Import the Libraries.  
·      Importing the dataset.  
·      Understanding Data Type and Summary of features.  
·      Take care of missing data  
·      Data Visualization.  
·      Drop the column from Data Frame & replace the missing value.  
·      Splitting the Dataset into Dependent and Independent variables  
·      Splitting Data into Train and Test.

 
Model Building:  
·      Training and testing the model  
·      Evaluation of Model  
·      Saving the Model  

Application Building:  
·      Create an HTML file  
·      Build a Python Code

Final UI:  
·      Dashboard Of the flask app.
## About the dataset

## Future scope

More accuracy can be gained by increasing the size of the dataset by generating synthetic data which can be obtained by scaling the applicant income , co-applicant income and loan amount columns and adding it to the existing dataset and running our model on the new dataset. Although, synthetic data generation is a very tedious and difficult process, it can help achieve a better accuracy.