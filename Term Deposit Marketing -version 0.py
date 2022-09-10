#!/usr/bin/env python

# ### Problem Statement

# Predict if the customer will subscribe (yes/no) to a term deposit (variable y)

# ### Data Description:
# 
# The data comes from direct marketing efforts of a European banking institution. The marketing campaign involves making a phone call to a customer, often multiple times to ensure a product subscription, in this case a term deposit. Term deposits are usually short-term deposits with maturities ranging from one month to a few years. The customer must understand when buying a term deposit that they can withdraw their funds only after the term ends. All customer information that might reveal personal information is removed due to privacy concerns.
# 
# #### Attributes:
# 
# + age : age of customer (numeric)
# 
# + job : type of job (categorical)
# 
# + marital : marital status (categorical)
# 
# + education (categorical)
# 
# + default: has credit in default? (binary)
# 
# + balance: average yearly balance, in euros (numeric)
# 
# + housing: has a housing loan? (binary)
# 
# + loan: has personal loan? (binary)
# 
# + contact: contact communication type (categorical)
# 
# + day: last contact day of the month (numeric)
# 
# + month: last contact month of year (categorical)
# 
# + duration: last contact duration, in seconds (numeric)
# 
# + campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# #### Output (desired target):
# 
# + y - has the client subscribed to a term deposit? (binary)
# 
# 
# #### Success Metric(s):
# 
# Hit 81% or above accuracy by evaluating with 5-fold cross validation and reporting the average performance score.
# 

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import data
data = pd.read_csv('term-deposit-marketing-2020.csv')

# Check the first 5 rows of our dataset
data.head()


# Check information
data.info()


# Now, we can see that we have 9 non-numerical features which we'll have to convert to be numeric. And by looking at the above, we can see that we don't have null values. 
# 
# To confirm that, we run the following code:
data.isnull().sum()

# Check the size of our data
data.shape


# Now, let us remove any duplicates should there be any.
data.drop_duplicates(inplace=True)

data.shape


# Le's convert categorical features to numerical.

from sklearn.preprocessing import LabelEncoder

# Get only categorical features
categorical_data = data.select_dtypes(exclude='int64')

# Drop the dependent feature
categorical_data.drop('y', axis=1, inplace=True)

encoder = LabelEncoder()


# encodes all the categorical columns
for i in categorical_data.columns:
    data[i] = encoder.fit_transform(data[i])


# Check how our new dataset now looks like
data.head()


# Now that we've converted all categorical to numericals, we can now scale our features to deal with the difference in scales for the features with higher magnitude to not govern our trained model, resulting in misinterpretation of data.
# 
# But first, let's get dependent and independant variables.
x = data.drop('y', axis=1)
y = data.y

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()

x = scaler.fit_transform(x)


# Now that we've scaled our data, let us get training and test data.

# #### Training and Test Data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Check the count of the classes in the dependent variable
y_train.value_counts()


# We can see that we are dealing with imbalance class problem, where there is a huge between our 2 classes. This needs to be dealt with to allow our trained model to have maximised accuracy and minimised errors.

sm = SMOTE(random_state=27)

# applying it to train set
x_train_smote, y_train_smote = sm.fit_resample(x_train, y_train)

# Check the count of the classes in the dependent variable, after over_sampling
y_train_smote.value_counts()


# Our dependent variable is now balanced. Now, we can proceed and train our models.


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# #### Logistic Regression

logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(x_train_smote, y_train_smote)
log_prediction = logmodel.predict(x_test)

# Score report
print(classification_report(y_test, log_prediction))


# #### Decision Tree

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(x_train_smote, y_train_smote)
dt_prediction = dt.predict(x_test)

# Score report
print(classification_report(y_test, dt_prediction))


# #### KNeighborsClassifier Model

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train_smote, y_train_smote)
knn_prediction = knn.predict(x_test)

# Score report
print(classification_report(y_test, knn_prediction))


# ### K-Fold Cross Validation

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


models = [('Logistic Regression: ', logmodel), ('Decision Tree: ', dt), ('K-Nearest Neigbors: ', knn)]

# Train k-fold 
results = []
for model_name, model in models:
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    cv_results = cross_val_score(i, x_train_smote, y_train_smote, cv=kfold, scoring='accuracy')
    each_result = "mean: {}, std: {}".format(np.round(cv_results.mean(),3), np.round(cv_results.std(),3))
    results.append([model_name, each_result])
    
# Check accuracy results
results


# We can see that all 3 of our models have an accuracy of 92% i.e. they all perform the same way. 

# We can see that k-fold cross validation has indeed improved the performance of our models, Logistic regression having moved from 85 to 92%,
# decision tree from 89 to 92%, and knn from 86 to 92% accuracies. 





