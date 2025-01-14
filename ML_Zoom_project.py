# %% [markdown]
# ## 1. Problem Description
# We are tasked with predicting the next day's closing price of Apple stock (AAPL) using historical stock data. The goal is to build a regression model to forecast future prices, leveraging features such as price change, moving averages, and volume changes. We will use both a Linear Regression model and a Random Forest Regressor to evaluate which model provides better predictions.
# 

# %% [markdown]
# ## 2. Exploratory Data Analysis (EDA)

# %%
import yfinance as yf
import pandas as pd


# AAPL stock data download
data = yf.download('TSLA', start='2014-01-01', end='2024-01-01')

#Resetting index to make 'Date' a column
data.reset_index(inplace=True)

#Display first few rows and check basic info
data.head()
data.info()
data.describe()


# %% [markdown]
# ## Feature Engineering

# %%
import matplotlib.pyplot as plt

# Calculate price change (difference between today's close and previous day's close)
data['Price_Change'] = data['Close'].diff()

# Calculate moving averages (5-day and 20-day)
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()

# Calculate volume change (difference between today's volume and the 5-day moving average of volume)
data['Volume_Change'] = data['Volume'] - data['Volume'].rolling(window=5).mean()

# Create the 'Next_Day_Close' column by shifting 'Close' by 1 day
data['Next_Day_Close'] = data['Close'].shift(-1)

# Drop rows with NaN values (required for moving averages and 'Next_Day_Close')
data.dropna(inplace=True)

# Visualize closing price over time
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.title('TSLA Stock Price (2014-2024)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()


# %% [markdown]
# ## 3. Model Training

# %%
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#features (X) and target (y)
X = data[['Price_Change', 'MA5', 'MA20', 'Volume_Change']] 
y = data['Next_Day_Close']  # Target is the next day's closing price



#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

#Initialize models
linear_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=500, random_state=42)

#Cross-validation
lr_cv_scores = cross_val_score(linear_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

#Train the models
linear_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


#Make predictions
lr_pred = linear_model.predict(X_test)
rf_pred = rf_model.predict(X_test)


print(f"Linear Regression CV Mean Squared Error: {-lr_cv_scores.mean()}")
print(f"Random Forest CV Mean Squared Error: {-rf_cv_scores.mean()}")



# %% [markdown]
# ## 4. Model Evaluation

# %%
from sklearn.model_selection import KFold
import numpy as np

#Initialize k-fold cross-validator
k = 5  
kf = KFold(n_splits=k, shuffle=False)  

# Initialize models
#linear_model = LinearRegression()
#rf_model = RandomForestRegressor(n_estimators=500, random_state=42)

# Step 2: Perform k-fold cross-validation
lr_mse_scores = []
lr_r2_scores = []
rf_mse_scores = []
rf_r2_scores = []

for train_index, val_index in kf.split(X):
    # Split the data into training and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train Linear Regression
    linear_model.fit(X_train, y_train)
    lr_val_pred = linear_model.predict(X_val)
    
    # Evaluate Linear Regression
    lr_mse_scores.append(mean_squared_error(y_val, lr_val_pred))
    lr_r2_scores.append(r2_score(y_val, lr_val_pred))
    
    # Train Random Forest
    rf_model.fit(X_train, y_train)
    rf_val_pred = rf_model.predict(X_val)
    
    # Evaluate Random Forest
    rf_mse_scores.append(mean_squared_error(y_val, rf_val_pred))
    rf_r2_scores.append(r2_score(y_val, rf_val_pred))

# Step 3: Calculate average metrics for each model
lr_avg_mse = np.mean(lr_mse_scores)
lr_avg_r2 = np.mean(lr_r2_scores)
rf_avg_mse = np.mean(rf_mse_scores)
rf_avg_r2 = np.mean(rf_r2_scores)

# Display results
print("Linear Regression k-Fold Cross-Validation Results:")
print(f"Average MSE: {lr_avg_mse:.2f}")
print(f"Average R²: {lr_avg_r2:.2f}")

print("\nRandom Forest k-Fold Cross-Validation Results:")
print(f"Average MSE: {rf_avg_mse:.2f}")
print(f"Average R²: {rf_avg_r2:.2f}")

# Step 8: Interpretation
print("\nInterpretation:")
if -lr_cv_scores.mean() < -rf_cv_scores.mean():
    print("Linear Regression has a lower mean squared error than Random Forest.")
else:
    print("Random Forest has a lower mean squared error than Linear Regression.")


# %% [markdown]
# ## 5. True vs Predicted Prices Visualization

# %%
# Step 11: Visualize the true vs predicted stock prices

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='True Prices', color='blue')
plt.plot(lr_pred, label='Linear Regression Predicted Prices', color='red', linestyle='--')
plt.plot(rf_pred, label='Random Forest Predicted Prices', color='green', linestyle='--')
plt.legend()
plt.title('True vs Predicted Stock Prices (Next Day)')
plt.show()


# %%
#Step 11: Make a prediction with a new data point
# Let's take the most recent day in the dataset (the last row in the dataset) for prediction.
latest_data = data.iloc[-1][['Price_Change', 'MA5', 'MA20', 'Volume_Change']].values.reshape(1, -1)

# Predict the next day's closing price using both models
lr_next_day_pred = linear_model.predict(latest_data)
rf_next_day_pred = rf_model.predict(latest_data)

print(f"Linear Regression Prediction for Next Day's Close: {lr_next_day_pred[0]}")
print(f"Random Forest Prediction for Next Day's Close: {rf_next_day_pred[0]}")


# %%
data.tail(5)

# %%
# Plot residuals to check for patterns
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate residuals
residuals = y_test - lr_pred

# Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='orange')
plt.title('Residuals of Linear Regression')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# %%
from sklearn.linear_model import Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Define a dictionary to hold the models and their hyperparameters
models = {
    'L1 (Lasso)': Lasso(alpha=0.1, random_state=42),
    'L2 (Ridge)': Ridge(alpha=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),  # l1_ratio = 0.5 combines L1 and L2
    'Stochastic (SGD)': SGDRegressor(penalty='l2', alpha=0.1, max_iter=1000, random_state=42)  # Default is L2 regularization
}

# Initialize lists to store results
results = []

# Loop through each model, fit, and evaluate
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    results.append({'Model': name, 'MSE': mse, 'R²': r2})

# Convert results to a DataFrame for easier comparison
results_df = pd.DataFrame(results)

# Display the results
print(results_df)


# %%
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Define a range of alpha values to tune
alphas = [0.01, 0.1, 1, 10, 100]

# Initialize a list to store results
ridge_results = []

# Loop through each alpha value
for alpha in alphas:
    # Initialize Ridge with the current alpha
    ridge = Ridge(alpha=alpha, random_state=42)
    
    # Fit the model on the training data
    ridge.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = ridge.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store the results
    ridge_results.append({
        'Alpha': alpha,
        'MSE': mse,
        'R²': r2
    })

# Convert results to a DataFrame for easier analysis
ridge_results_df = pd.DataFrame(ridge_results)

# Sort results by MSE
ridge_results_df = ridge_results_df.sort_values(by='MSE').reset_index(drop=True)

# Display the results
print(ridge_results_df)

# %%
from sklearn.model_selection import cross_val_score




#Initialize a list to store cross-validation results
cv_results = []

# Step 3: Perform cross-validation for each alpha value
#for alpha in alphas:
    # Initialize Ridge Regression with the current alpharidge = Ridge(alpha=0.01, random_state=42)
    
    # Perform 5-fold cross-validation
cv_mse = cross_val_score(ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_r2 = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
    
    # Calculate the mean of the cross-validation scores
mean_mse = -cv_mse.mean()  # Convert negative MSE to positive
mean_r2 = cv_r2.mean()
    
    # Store the results
cv_results.append({
        'Alpha': 0.01,
        'Mean CV MSE': mean_mse,
        'Mean CV R²': mean_r2
    })

# Convert results to a DataFrame for easier analysis
cv_results_df = pd.DataFrame(cv_results)

# Sort results by Mean CV MSE
cv_results_df = cv_results_df.sort_values(by='Mean CV MSE').reset_index(drop=True)

# Display the cross-validation results
print(cv_results_df)


# %%
#Select the best alpha
best_alpha = cv_results_df.loc[0, 'Alpha']

#Train the Ridge model with the best alpha
ridge_model = Ridge(alpha=best_alpha, random_state=42)
ridge_model.fit(X_train, y_train)

#Make predictions on the test set
ridge_predictions = ridge_model.predict(X_test)

#Evaluate the model on the test set
ridge_mse = mean_squared_error(y_test, ridge_predictions)
ridge_r2 = r2_score(y_test, ridge_predictions)

print(f"Best Alpha: {best_alpha}")
print(f"Test Set MSE: {ridge_mse}")
print(f"Test Set R²: {ridge_r2}")


# %%
import matplotlib.pyplot as plt

# Plot true vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(ridge_predictions, label='Predicted Prices (Ridge)', color='red', linestyle='--')
plt.title('Ridge Regression: Actual vs Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()


# %%
# Step 11: Make a prediction with a new data point (example)
# Take the most recent day in the dataset (second-to-last row) for prediction
latest_data = data.iloc[-2][['Price_Change', 'MA5', 'MA20', 'Volume_Change']].values.reshape(1, -1)

# Predict the next day's closing price using the trained Ridge model
ridge_next_day_pred = ridge_model.predict(latest_data)

print(f"Ridge Regression Prediction for Next Day's Close: {ridge_next_day_pred[0]}")


# %%
# Predict using Linear Regression
lr_next_day_pred = linear_model.predict(latest_data)

# Predict using Random Forest
rf_next_day_pred = rf_model.predict(latest_data)

# Print predictions
print(f"Linear Regression Prediction for Next Day's Close: {lr_next_day_pred[0]}")
print(f"Random Forest Prediction for Next Day's Close: {rf_next_day_pred[0]}")
print(f"Ridge Regression Prediction for Next Day's Close: {ridge_next_day_pred[0]}")


# %%



# %%



