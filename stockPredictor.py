import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
"""
Dataset:
Yahoo Finance. (2025). *Tesla, Inc. (TSLA) Stock Historical Data*. Retrieved from [https://finance.yahoo.com/quote/TSLA/history](https://finance.yahoo.com/quote/TSLA/history)
"""


# number of past days to use as input (60 past days of data)
daysOfData = 60

#download past stock data
stock = "TSLA"
df = yf.download(stock, start="2015-01-01", end="2025-01-01")

#get 'Close' prices
data = df[['Close']].values

#Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

#prepare X (the past days), and y (the next day)
X, y = [], []
for i in range(len(data) - daysOfData):
    X.append(data[i : i + daysOfData])
    y.append(data[i + daysOfData])

X, y = np.array(X), np.array(y)

#split into training (80%) and testing (20%)
XTrain, XTest = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
yTrain, yTest = y[:int(len(X) * 0.8)], y[int(len(X) * 0.8):]

#reshape the data for LSTM
XTrain = XTrain.reshape(XTrain.shape[0], XTrain.shape[1], 1)
XTest = XTest.reshape(XTest.shape[0], XTest.shape[1], 1) 


#create LSTM model
model = Sequential([
    Input(shape = (daysOfData, 1)),
    LSTM(64, return_sequences = True), 
    LSTM(64),
    Dense(32), 
    Dense(1)
])

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#create early stopping callback
earlyStopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)


#train
model.fit(XTrain, yTrain, epochs = 100, batch_size=32, validation_data=(XTest, yTest), callbacks = [earlyStopping])

# Predict
predictions = model.predict(XTest)

predictions = predictions.reshape(predictions.shape[0], 1) 

predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

# Plotting the real vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(yTest):], scaler.inverse_transform(np.array(yTest).reshape(-1, 1)), label='Real Prices')
plt.plot(df.index[-len(yTest):], predictions, label='Predicted Prices')
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

