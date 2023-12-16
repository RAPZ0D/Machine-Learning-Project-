import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

tf.__version__

# Load the dataset
file_path = 'Churn_Modelling.csv'
data = pd.read_csv(file_path)

# Label Encoding for 'Gender'
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# One-Hot Encoding for 'Geography'
onehot_encoder = OneHotEncoder(drop='first', sparse=False)
geography_encoded = onehot_encoder.fit_transform(data[['Geography']])
geography_df = pd.DataFrame(geography_encoded, columns=[f'Geography_{i}' for i in range(geography_encoded.shape[1])])
data = pd.concat([data, geography_df], axis=1)
data.drop('Geography', axis=1, inplace=True)


# Splitting the dataset into features and target
X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Building the ANN
ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])


# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Training the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))


# Assuming 'ann' is the trained ANN model
predictions = ann.predict(X_test)

# Convert predictions to binary (0 or 1) based on a threshold (e.g., 0.5)
binary_predictions = (predictions > 0.5).astype(int)

# Display the predictions
print("Predictions:")
print(binary_predictions)

