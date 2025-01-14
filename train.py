import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and preprocess data
import yfinance as yf
import pandas as pd
# AAPL stock data download
data = yf.download('TSLA', start='2014-01-01', end='2024-01-01')
#Resetting index to make 'Date' a column
data.reset_index(inplace=True)
data['Price_Change'] = data['Close'].diff()
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['Volume_Change'] = data['Volume'] - data['Volume'].rolling(window=5).mean()
data['Next_Day_Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Features and target
X = data[['Price_Change', 'MA5', 'MA20', 'Volume_Change']]
y = data['Next_Day_Close']

print(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(linear_model, 'linear_model.pkl')
print("Linear Regression model saved as 'linear_model.pkl'")
