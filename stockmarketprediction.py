import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
from scipy.stats import randint, uniform  # Added import for randint and uniform
from sklearn.model_selection import RandomizedSearchCV  # Added import for RandomizedSearchCV

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(data, window):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)


    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

# Function to calculate MACD, Signal Line, and MACD Histogram
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['ShortEMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['LongEMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['ShortEMA'] - data['LongEMA']
    data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['MACD Histogram'] = data['MACD'] - data['Signal Line']

# Load the training data
train_data = pd.read_csv("/kaggle/input/ue21cs342aa2/train.csv")

# Convert 'Date' column to datetime
train_data['Date'] = pd.to_datetime(train_data['Date'])

# Set alpha (smoothing factor)
alpha = 0.2

# Initialize the forecast with the first observed close value
forecast = train_data["Close"].iloc[-1]

# Load the test data
test_data = pd.read_csv("/kaggle/input/ue21cs342aa2/test.csv")

# Convert 'Date' column to datetime
test_data['Date'] = pd.to_datetime(test_data['Date'])

# Initialize lists to store predictions
predicted_closes = []

# Iterate over test data to make SARIMA predictions
for i in range(len(test_data)):
    close_value = alpha * test_data["Open"].iloc[i] + (1 - alpha) * forecast
    forecast = close_value

    # Manually tune and fit the SARIMA model (you should tune the hyperparameters)
    sarima_model = SARIMAX(train_data["Close"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_results = sarima_model.fit(disp=False)

    sarima_forecast = sarima_results.get_forecast(steps=1)
    sarima_close = sarima_forecast.predicted_mean.iat[0]

    if np.abs(sarima_close - close_value) < 1.0:
        predicted_close = sarima_close
    else:
        predicted_close = close_value

    predicted_closes.append(predicted_close)

test_data["Close"] = predicted_closes

# Define your threshold for Buy and Sell
threshold_buy = 0.0
threshold_sell = 0.0

train_data['MA'] = train_data['Close'].rolling(window=10).mean()
test_data['MA'] = test_data['Close'].rolling(window=10).mean()
window = 14
train_data['RSI'] = calculate_rsi(train_data['Close'], window)
test_data['RSI'] = calculate_rsi(test_data['Close'], window)

# Calculate MACD features for training and test data
calculate_macd(train_data)
calculate_macd(test_data)

label_encoder = LabelEncoder()
train_data["Strategy_Label"] = label_encoder.fit_transform(train_data["Strategy"])

X_train = train_data[["Open", "Close", "Volume", "MA", "RSI", "MACD", "Signal Line", "MACD Histogram"]]
X_test = test_data[["Open", "Close", "Volume", "MA", "RSI", "MACD", "Signal Line", "MACD Histogram"]]
y_train = train_data["Strategy_Label"]

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define a distribution of hyperparameters to sample from for XGBoost
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
}

# Create an XGBoost model
model = xgb.XGBClassifier(random_state=42)

# Perform random search for XGBoost hyperparameters
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Train the best model on the full training data
best_model.fit(X_train, y_train)

# Evaluate the best model on the validation set
accuracy = best_model.score(X_val, y_val)
print("Best Model - Accuracy on Validation Set:", accuracy)

# Predict the strategy for the test data using the best model
predicted_strategy = best_model.predict(X_test)

# Decode the numerical strategy labels back to "Buy," "Sell," or "Hold"
predicted_strategy = label_encoder.inverse_transform(predicted_strategy)

# Add the predicted strategy to the test data
test_data["Strategy"] = predicted_strategy

# Save the DataFrame to a CSV file named "submission.csv"
submission_df = test_data[['id', "Date", "Close", "Strategy"]]
submission_df.to_csv("submission.csv", index=False)