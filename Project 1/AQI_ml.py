import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

df = pd.read_csv("AQI By State 1980-2022.csv")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df.head()

X = df.iloc[:,3:].values
y = df.iloc[:,2].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

y
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled X_train data:")
print(X_train_scaled)
print("\nScaled X_test data:")
print(X_test_scaled)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a linear regression model
model = LinearRegression()

# Fit the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
predictions = model.predict(X_test_scaled)

# Evaluate the model (for example, using mean squared error)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, precision_score
# Create a logistic regression model with increased max_iter
classifier = LogisticRegression(max_iter=1000)  # Increase max_iter as needed

# Fit the model on the scaled training data
classifier.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
predictions = classifier.predict(X_test_scaled)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)

# Calculate and print precision
# Assuming 'average' parameter as 'weighted' for multi-class classification
precision = precision_score(y_test, predictions, average='weighted')
print("\nPrecision:", precision)