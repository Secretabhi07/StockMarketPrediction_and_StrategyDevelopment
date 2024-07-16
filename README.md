# Stock Price Prediction and Strategy Development Using Machine Learning

This project involves the development of a sophisticated stock price prediction system utilizing SARIMA and XGBoost models, complemented by key technical indicators. The primary objective is to accurately predict stock closing prices and generate actionable trading strategies ("Buy," "Sell," "Hold").

## Key Features

1. **Time Series Analysis with SARIMA:**
   - Implemented SARIMA (Seasonal Autoregressive Integrated Moving Average) for time series forecasting.
   - Tuned model parameters to fit the historical stock price data.

2. **Machine Learning with XGBoost:**
   - Used XGBoost, a powerful gradient boosting algorithm, to predict stock prices.
   - Conducted hyperparameter tuning with RandomizedSearchCV to optimize model performance.

3. **Technical Indicators:**
   - Calculated Relative Strength Index (RSI) to identify overbought or oversold conditions.
   - Implemented Moving Average Convergence Divergence (MACD) for trend-following momentum.

4. **Data Preprocessing:**
   - Loaded and processed historical stock price data.
   - Handled missing values, outliers, and other data quality issues.

5. **Feature Engineering:**
   - Created additional features such as moving averages and MACD histogram.
   - Prepared datasets for training and testing machine learning models.

6. **Model Evaluation:**
   - Split data into training and validation sets.
   - Evaluated model performance using accuracy metrics on validation data.

7. **Trading Strategy Prediction:**
   - Predicted trading strategies based on model insights.
   - Transformed numerical labels into actionable "Buy," "Sell," "Hold" recommendations.

8. **Results:**
   - Achieved high accuracy in stock price prediction.
   - Demonstrated effective strategy generation for trading decisions.

## Technologies Used

- **Python:** Pandas, NumPy, Scikit-learn, XGBoost
- **Time Series Analysis:** Statsmodels (SARIMAX)
- **Machine Learning:** RandomizedSearchCV for hyperparameter tuning
- **Technical Indicators:** Custom functions for RSI and MACD calculations
