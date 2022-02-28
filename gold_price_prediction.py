# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Data Collection and Processing
gold_data = pd.read_csv("gold_price_raw_data.csv")

# print first 5 rows in the dataframe
print(gold_data.head())

# print last 5 rows of the dataframe
print(gold_data.tail())

# number of rows and columns
print(gold_data.shape)

# getting some basic informations about the data
gold_data.info()

# checking the number of missing values
print(gold_data.isnull().sum())

# getting the statistical measures of the data
print(gold_data.describe())

# Correlation which has both positive and negative correalation
correlation = gold_data.corr()

# constructing a heatmap to understand the correlatiom
plt.figure(figsize = (8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True, annot_kws={'size':8}, cmap='Blues')

# correlation values of GLD
print(correlation['GLD'])

# checking the distribution of the GLD Price
sns.distplot(gold_data['GLD'],color='green')


# Splitting the Features and Target
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']

print(X)
print(Y)

# Splitting into Training data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

# Model Training: Random Forest Regressor
regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(X_train,Y_train)

# prediction on Test Data
test_data_prediction = regressor.predict(X_test)

print(test_data_prediction)

# R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)


# Compare the Actual Values and Predicted Values in a Plot
Y_test = list(Y_test)

plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()
