import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the data from the CSV file
file_path = 'infolimpioavanzadoTarget.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Assuming 'Date' is in datetime format, if not, convert it
data['date'] = pd.to_datetime(data['date'])

# Sort the data by date
data.sort_values(by='Date', inplace=True)

# Create a new feature for days since the beginning
data['Days'] = (data['date'] - data['date'].min()).dt.days

# Use 'Days' as the independent variable and 'Close' as the dependent variable
X = data[['Days']]
y = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Visualize the results
plt.scatter(X_test, y_test, color='black', label='Actual Prices')
plt.plot(X_test, predictions, color='blue', linewidth=3, label='Predicted Prices')
plt.title('Stock Price Prediction with Linear Regression')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
# Import necessary libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
# import warnings
# warnings.filterwarnings("ignore")
# df = pd.read_csv('infolimpioavanzadoTarget.csv')

# # Exploratory Data Analysis (EDA)
# print(df.info())
# print(df.describe())

# # Visualize key statistics and trends in stock prices
# plt.figure(figsize=(10, 6))
# sns.lineplot(x='Date', y='Close', data=df)
# plt.title('Stock Prices Over Time')
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.show()

# # Predictive Modeling

# # Preprocess the data
# # Assuming 'Close' is the target variable
# X = df.drop(['Close', 'Date'], axis=1)
# y = df['Close']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Linear Regression Model
# linear_model = LinearRegression()
# linear_model.fit(X_train, y_train)

# # Make predictions on the test set
# linear_predictions = linear_model.predict(X_test)

# # Evaluate the model
# linear_mse = mean_squared_error(y_test, linear_predictions)
# print(f'Linear Regression Mean Squared Error: {linear_mse}')

# # Random Forest Regressor Model with Hyperparameter Tuning
# rf_model = RandomForestRegressor()

# # Define hyperparameters for tuning
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # Use GridSearchCV for hyperparameter tuning
# grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_params = grid_search.best_params_

# # Train the model with the best parameters
# best_rf_model = RandomForestRegressor(**best_params)
# best_rf_model.fit(X_train, y_train)

# # Make predictions on the test set
# rf_predictions = best_rf_model.predict(X_test)

# # Evaluate the tuned model
# rf_mse = mean_squared_error(y_test, rf_predictions)
# print(f'Random Forest Regressor Mean Squared Error: {rf_mse}')

