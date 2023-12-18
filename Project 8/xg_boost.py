import pandas as pd

# Load the CSV file into a DataFrame
data = pd.read_csv('Data.csv')

# Display the first few rows to understand the structure
print(data.head())


# Separating features and target variable
X = data.drop('Class', axis=1)
y = data['Class']


from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import xgboost as xgb
# Map the class labels to [0, 1]
y_train_mapped = y_train.map({2: 0, 4: 1})
y_test_mapped = y_test.map({2: 0, 4: 1})

# Initialize XGBoost classifier
xgb_clf = xgb.XGBClassifier(random_state=42)

# Train the classifier on the training data with the mapped labels
xgb_clf.fit(X_train, y_train_mapped)


# Predict on the test set
y_pred = xgb_clf.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")


import xgboost as xgb
import matplotlib.pyplot as plt

# Assuming xgb_clf is already trained
# Get feature importance scores
feature_importance = xgb_clf.feature_importances_
feature_names = X.columns

# Sort feature importances in descending order
sorted_idx = feature_importance.argsort()[::-1]

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.xticks(range(X.shape[1]), feature_names[sorted_idx], rotation=90)
plt.xlabel('Feature Importance')
plt.title('Feature Importance Plot')
plt.show()
