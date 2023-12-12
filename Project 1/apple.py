import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('aapl.csv')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Get a summary of the dataset
print("\nDataset summary:")
print(data.info())

# Display basic statistics of the numerical columns
print("\nBasic statistics of numerical columns:")
print(data.describe())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare the data for regression
X = data[['Open', 'High', 'Low', 'Volume', 'Adj Close']]  # Exclude 'Close' from features
y = data['Close']  # Predicting the 'Close' stock price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust test_size as needed

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)

import matplotlib.pyplot as plt

# Assuming 'predictions' and 'y_test' are obtained from the regression model

plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, color='blue', label='Actual vs. Predicted', alpha=0.5)
plt.plot(y_test, y_test, color='red', label='Ideal Prediction', linestyle='--')  # Plotting ideal prediction line (y = x)
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()


residuals = y_test - predictions

plt.figure(figsize=(8, 6))
plt.scatter(y_test, residuals, color='green')
plt.title('Residual Plot')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()

import seaborn as sns
import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('aapl.csv')

# Display pairwise relationships with a pairplot
sns.pairplot(data)
plt.show()

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for a specific year and month (e.g., 2008-10)
specific_date = '2008-10-14'
year_month_data = data[data['Date'].dt.strftime('%Y-%m') == specific_date[:7]]

# Plotting the line plot for the specific year and month
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Close', data=year_month_data)
plt.title(f'Apple Stock Closing Prices for {specific_date[:7]}')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Exclude the 'Date' column for the heatmap
numerical_data = data.drop('Date', axis=1)

# Create a heatmap of correlations
plt.figure(figsize=(10, 8))
correlation = numerical_data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Extracting year from the date
data['Year'] = data['Date'].dt.year

# Grouping data by year and finding the highest and lowest values for each year
highest_values = data.groupby('Year')['High'].max()
lowest_values = data.groupby('Year')['Low'].min()

# Plotting the highest and lowest values for each year
plt.figure(figsize=(12, 6))

sns.lineplot(x=highest_values.index, y=highest_values.values, color='red', label='Highest Value')
sns.lineplot(x=lowest_values.index, y=lowest_values.values, color='blue', label='Lowest Value')

plt.title('Highest and Lowest Apple Stock Values Each Year')
plt.xlabel('Year')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.show()

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the specified date ('2008-10-14')
specific_date = '2008-10-14'
date_data = data[data['Date'].dt.strftime('%Y-%m-%d') == specific_date]

# Plotting the Open and Close prices
plt.figure(figsize=(8, 6))
sns.barplot(x=['Open', 'Close'], y=[date_data['Open'].values[0], date_data['Close'].values[0]], palette='coolwarm')
plt.title(f'Open and Close Prices for {specific_date}')
plt.ylabel('Stock Price')
plt.show()

# Plotting the High and Low prices using error bars
plt.figure(figsize=(8, 6))
plt.errorbar(x=['High', 'Low'], y=[date_data['High'].values[0], date_data['Low'].values[0]],
             yerr=[[date_data['Open'].values[0] - date_data['Low'].values[0], date_data['High'].values[0] - date_data['Close'].values[0]]], fmt='o', color='black')
plt.title(f'High and Low Prices for {specific_date}')
plt.ylabel('Stock Price')
plt.show()

# Load the data from the CSV file
data = pd.read_csv('aapl.csv')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the year 2008
data_2008 = data[data['Date'].dt.year == 2008]

# Set 'Date' column as the index
data_2008.set_index('Date', inplace=True)

# Convert relevant columns to numeric if they aren't already
data_2008[['Open', 'High', 'Low', 'Close']] = data_2008[['Open', 'High', 'Low', 'Close']].apply(pd.to_numeric, errors='coerce')

# Compute rolling average for the stock prices in 2008
rolling_avg_2008 = data_2008[['Open', 'High', 'Low', 'Close']].rolling(7).mean()

# Set the seaborn style
sns.set_theme(style="whitegrid")

# Plotting the rolling average for Apple stock prices in 2008
plt.figure(figsize=(10, 6))
sns.lineplot(data=rolling_avg_2008, palette="tab10", linewidth=2.5)

plt.title('Rolling 7-Day Average of Apple Stock Prices in 2008')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(title='Stock', loc='upper left', labels=['Open', 'High', 'Low', 'Close'])  # Adjust labels if needed
plt.show()


