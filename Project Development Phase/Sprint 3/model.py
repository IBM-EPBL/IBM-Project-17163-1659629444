import pandas as pd
from matplotlib import pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore")

loan_data = pd.read_csv(r'LoanDataset.csv')


"""## EDA

## Uni variate Analysis
"""

plt.figure(figsize=(20, 12))
plt.subplot(231)
sns.distplot(loan_data['ApplicantIncome'], color='r')
plt.subplot(232)
sns.distplot(loan_data['Credit_History'])
plt.subplot(233)
sns.distplot(loan_data['CoapplicantIncome'], color='r')
plt.subplot(234)
sns.distplot(loan_data['LoanAmount'])
plt.subplot(235)
sns.distplot(loan_data['Loan_Amount_Term'], color='r')
# plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(121)
sns.countplot(loan_data['Gender'])
plt.subplot(122)
sns.countplot(loan_data['Education'])
# plt.show()

plt.figure(figsize=(20, 5))
plt.subplot(131)
sns.countplot(loan_data['Gender'], hue=loan_data['Education'])
plt.subplot(132)
sns.countplot(loan_data['Married'], hue=loan_data['Gender'])
plt.subplot(133)
sns.countplot(loan_data['Self_Employed'], hue=loan_data['Education'])
# plt.show()
plt.figure(figsize=(20, 10))
sns.countplot(loan_data['Property_Area'], hue=loan_data['Loan_Amount_Term'])
# plt.show()

temp_data = loan_data.drop(
    ['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Loan_Status'], axis=1)
cols = temp_data.columns
for c in cols:
    plt.figure(c, figsize=(7, 4))
    sns.countplot(temp_data[c], hue=loan_data['Loan_Status'])

# plt.show()

loan_data['ApplicantIncome'].max()
maximum_income = loan_data['ApplicantIncome'].unique().max()
minimum_income = loan_data['ApplicantIncome'].unique().min()
print(maximum_income, minimum_income)

loan_data['CoapplicantIncome'].max()
maximum_income = loan_data['CoapplicantIncome'].unique().max()
minimum_income = loan_data['CoapplicantIncome'].unique().min()
print(maximum_income, minimum_income)

copy_data = loan_data.copy()
bin_range = [150, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 81000]
copy_data['Income_group'] = pd.cut(x=copy_data['ApplicantIncome'], bins=bin_range)
copy_data['Income_group'].value_counts()

plt.figure(figsize=(20, 5))
sns.countplot(x=copy_data['Income_group'], hue=copy_data['Loan_Status'])
# plt.show()

bin_range = [1, 10000, 20000, 30000, 42000]
copy_data['CoIncome_group'] = pd.cut(x=copy_data['CoapplicantIncome'], bins=bin_range)
copy_data['CoIncome_group'].value_counts()

plt.figure(figsize=(20, 5))
sns.countplot(x=copy_data['CoIncome_group'], hue=copy_data['Loan_Status'])
# plt.show()

"""### Multivariate Analysis"""

sns.swarmplot(loan_data['Gender'], loan_data['ApplicantIncome'], hue=loan_data['Loan_Status'])

"""### Descriptive Analysis"""

loan_data.describe()

stat_data = loan_data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']]
for i in stat_data:
    plt.hist(loan_data[i])
    # plt.show()

"""## Data Preprocessing"""

loan_data = loan_data.drop(columns=['Loan_ID'], axis=1)
loan_data.head()

"""## Handling Missing Values"""

loan_data.isna()

null_data = loan_data.isna().sum()
null_data.sort_values()

msno.bar(loan_data)

msno.matrix(loan_data)

unique_cols = loan_data.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'], axis=1)
for col in unique_cols:
    print(col, loan_data[col].unique())

loan_data['Gender'] = loan_data['Gender'].fillna(loan_data['Gender'].mode()[0])
loan_data['Married'] = loan_data['Married'].fillna(loan_data['Married'].mode()[0])
loan_data['Dependents'] = loan_data['Dependents'].str.replace('+', '')
loan_data['Self_Employed'] = loan_data['Self_Employed'].fillna(loan_data['Self_Employed'].mode()[0])
loan_data['LoanAmount'] = loan_data['LoanAmount'].fillna(loan_data['LoanAmount'].mode()[0])
loan_data['Loan_Amount_Term'] = loan_data['Loan_Amount_Term'].fillna(loan_data['Loan_Amount_Term'].mode()[0])
loan_data['Credit_History'] = loan_data['Credit_History'].fillna(loan_data['Credit_History'].mode()[0])
# loan_data

loan_data['CoapplicantIncome'] = loan_data['CoapplicantIncome'].astype('int64')
loan_data['LoanAmount'] = loan_data['LoanAmount'].astype('int64')
loan_data['Loan_Amount_Term'] = loan_data['Loan_Amount_Term'].astype('int64')
loan_data['Credit_History'] = loan_data['Credit_History'].astype('int64')
# loan_data

label_encoder = preprocessing.LabelEncoder()
label_encoding_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area',
                          'Loan_Status']
for col in label_encoding_columns:
    loan_data[col] = label_encoder.fit_transform(loan_data[col])

# loan_data

smk = SMOTETomek(0.90)

y = loan_data['Loan_Status']
x = loan_data.drop(columns=['Loan_Status'], axis=1)

x_bal, y_bal = smk.fit_resample(x, y)
print(y.value_counts())
print(y_bal.value_counts())

sc = StandardScaler()
x_bal = sc.fit_transform(x_bal)
x_bal = pd.DataFrame(x_bal)

# loan_data

X_train, X_test, y_train, y_test = train_test_split(x_bal, y_bal, test_size=0.33, random_state=42)

"""## MODEL BUILDING"""

acc = {}


def Decision_Tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # predicting the outcome
    y_pred = dt.predict(X_test)
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    # plt.show()
    print("Classification Report")
    cr = classification_report(y_test, y_pred)
    print(cr)
    acc['Decision Tree Classifier'] = accuracy_score(y_test, y_pred)


Decision_Tree(X_train, X_test, y_train, y_test)


def Random_Forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    # predicting the outcome
    y_pred = rf.predict(X_test)
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    # plt.show()
    print("Classification Report")
    cr = classification_report(y_test, y_pred)
    print(cr)
    acc['Random Forest Classifier'] = accuracy_score(y_test, y_pred)


Random_Forest(X_train, X_test, y_train, y_test)


def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    # predicting the outcome
    y_pred = knn.predict(X_test)
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    # plt.show()
    print("Classification Report")
    cr = classification_report(y_test, y_pred)
    print(cr)
    acc['KNN'] = accuracy_score(y_test, y_pred)


KNN(X_train, X_test, y_train, y_test)


def XGboost(X_train, X_test, y_train, y_test):
    xg = GradientBoostingClassifier()
    xg.fit(X_train, y_train)

    # predicting the outcome
    y_pred = xg.predict(X_test)
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    # plt.show()
    print("Classification Report")
    cr = classification_report(y_test, y_pred)
    print(cr)
    acc['Gradient Boost'] = accuracy_score(y_test, y_pred)


XGboost(X_train, X_test, y_train, y_test)
xg = GradientBoostingClassifier()
xg.fit(X_train, y_train)

# input_feature = [1,1,1,1,1,5849,1508,120,360,1,1]
input_feature = ['Female', 'Yes', 0, 'Graduate', 'Yes', 584900, 0, 120, 360, 1, 'Urban']
input_feature = [np.array(input_feature)]
print(input_feature)
names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
         'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
data = pd.DataFrame(input_feature, columns=names)
print(data)
new = data.copy()
label_encoder = preprocessing.LabelEncoder()
label_encoding_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in label_encoding_columns:
    new[col] = label_encoder.fit_transform(new[col])
preed = xg.predict(new)
print(preed)
print(acc)

# import pickle
#
# xg = GradientBoostingClassifier()
# xg.fit(X_train, y_train)
# # save the model to disk
# pickle.dump(xg, open('model3.pkl','wb'))
#
# model = pickle.load(open('model3.pkl','rb'))
#
# print(model.predict(X_test))
