import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

data = pd.read_csv("Social_Network_Ads.csv")

data.head()

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values 

X

y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

X_train_scaled

X_test_scaled

# K nearest neighbours 
from sklearn.neighbors import KNeighborsClassifier

# Initializing the KNN classifier with multiple parameters
k = 5  # Number of neighbors
weights = 'uniform'  # Weight function used in prediction: 'uniform' or 'distance'
algorithm = 'auto'  # Algorithm used to compute the nearest neighbors: 'auto', 'ball_tree', 'kd_tree', 'brute'
metric = 'euclidean'  # Distance metric for calculating neighbors: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', etc.

knn = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm, metric=metric)

# Training the KNN model on the scaled training data
knn.fit(X_train_scaled, y_train)

# Making predictions on the scaled test data
predictions = knn.predict(X_test_scaled)

from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Calculate precision
precision = precision_score(y_test, predictions)
print(f"Precision: {precision:.4f}")

# Calculate recall
recall = recall_score(y_test, predictions)
print(f"Recall: {recall:.4f}")

from sklearn.neighbors import KNeighborsClassifier

# Creating an instance of KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can change the number of neighbors here

# Fitting the classifier to the scaled training data
knn.fit(X_train_scaled, y_train)  # Assuming you have X_train_scaled and y_train

# Visualizing decision boundaries
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plotting decision boundaries
h = 0.1  # step size in the mesh
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plotting training points with labeled scatter plots for each class
for i, c in zip(range(len(np.unique(y_train))), cmap_bold.colors):
    plt.scatter(X_train_scaled[y_train == i, 0], X_train_scaled[y_train == i, 1],
                c=[c], cmap=cmap_bold, edgecolor='k', s=60, label=f'Class {i}')

plt.xlabel('Feature 1 Scaled')
plt.ylabel('Feature 2 Scaled')
plt.title('KNN Classification (K=5) - Decision Boundaries')
plt.legend()
plt.show()

# Support Vector Machines 
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
# Creating an instance of SVM classifier
svm = SVC(kernel='linear', C=1.0)  # You can choose different kernels and adjust C value

# Fitting the classifier to the scaled training data
svm.fit(X_train_scaled, y_train)  # Assuming you have X_train_scaled and y_train

# Making predictions on the test set
y_pred = svm.predict(X_test_scaled)  # Assuming you have X_test_scaled

# Calculating accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


# Visualizing decision boundaries with different colors
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plotting decision boundaries
h = 0.1  # step size in the mesh
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Define colors for test points
colors = ['#FF4500', '#228B22', '#0000CD']  # You can add more colors if needed

# Plotting test points with labeled scatter plots for each class using defined colors
for i, c in zip(range(len(np.unique(y_test))), colors):
    plt.scatter(X_test_scaled[y_test == i, 0], X_test_scaled[y_test == i, 1],
                c=c, edgecolor='k', s=60, label=f'Class {i}')

plt.xlabel('Feature 1 Scaled')
plt.ylabel('Feature 2 Scaled')
plt.title('SVM Classification - Decision Boundaries')
plt.legend()
plt.show()

# Naive Bayes 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Creating an instance of Naive Bayes classifier
nb = GaussianNB()

# Fitting the classifier to the scaled training data
nb.fit(X_train_scaled, y_train)  # Assuming you have X_train_scaled and y_train

# Making predictions on the test set
y_pred_nb = nb.predict(X_test_scaled)  # Assuming you have X_test_scaled

# Calculating accuracy, precision, and recall
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')

print(f"Naive Bayes - Accuracy: {accuracy_nb:.4f}")
print(f"Naive Bayes - Precision: {precision_nb:.4f}")
print(f"Naive Bayes - Recall: {recall_nb:.4f}")

# Visualizing decision boundaries
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plotting decision boundaries
h = 0.1  # step size in the mesh
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plotting test points with predicted classes
scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_nb, cmap=cmap_bold, edgecolor='k', s=60)

# Creating a legend for the scatter plot
legend = plt.legend(*scatter.legend_elements(), title='Classes', loc='upper right')
plt.xlabel('Feature 1 Scaled')
plt.ylabel('Feature 2 Scaled')
plt.title('Naive Bayes Classification - Decision Boundaries and Test Points')
plt.show()

# Decision Tree Classifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Creating an instance of Decision Tree classifier
dt = DecisionTreeClassifier()

# Fitting the classifier to the scaled training data
dt.fit(X_train_scaled, y_train)  # Assuming you have X_train_scaled and y_train

# Making predictions on the test set
y_pred_dt = dt.predict(X_test_scaled)  # Assuming you have X_test_scaled

# Calculating accuracy, precision, and recall
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')

print(f"Decision Tree - Accuracy: {accuracy_dt:.4f}")
print(f"Decision Tree - Precision: {precision_dt:.4f}")
print(f"Decision Tree - Recall: {recall_dt:.4f}")

# Visualizing decision boundaries
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plotting decision boundaries
h = 0.1  # step size in the mesh
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plotting test points with predicted classes
scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_dt, cmap=cmap_bold, edgecolor='k', s=60)

# Creating a legend for the scatter plot
legend = plt.legend(*scatter.legend_elements(), title='Classes', loc='upper right')
plt.xlabel('Feature 1 Scaled')
plt.ylabel('Feature 2 Scaled')
plt.title('Decision Tree Classification - Decision Boundaries and Test Points')
plt.show()


# Randon Forest Classifer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Creating an instance of Random Forest classifier
rf = RandomForestClassifier()

# Fitting the classifier to the scaled training data
rf.fit(X_train_scaled, y_train)  # Assuming you have X_train_scaled and y_train

# Making predictions on the test set
y_pred_rf = rf.predict(X_test_scaled)  # Assuming you have X_test_scaled

# Calculating accuracy, precision, and recall
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')

print(f"Random Forest - Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest - Precision: {precision_rf:.4f}")
print(f"Random Forest - Recall: {recall_rf:.4f}")

# Visualizing decision boundaries
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Plotting decision boundaries
h = 0.1  # step size in the mesh
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plotting test points with predicted classes
scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_rf, cmap=cmap_bold, edgecolor='k', s=60)

# Creating a legend for the scatter plot
legend = plt.legend(*scatter.legend_elements(), title='Classes', loc='upper right')
plt.xlabel('Feature 1 Scaled')
plt.ylabel('Feature 2 Scaled')
plt.title('Random Forest Classification - Decision Boundaries and Test Points')
plt.show()

