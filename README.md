# Tesla Stock Price Prediction Using LSTM (Python)

## Description

This project uses an LSTM (Long Short-Term Memory) model to predict the stock price of Tesla (TSLA). The model is trained using past closing price data to predict future stock prices, giving insights into stock price trends.

## Dataset

The dataset used in this project is sourced from Yahoo Finance and contains historical stock data of Tesla (TSLA) from January 1, 2015, to January 1, 2025. It includes features, such as the **Closing price**, which is the primary feature used for training the model.

## Libraries Used

The following libraries were used in the development of this project:
- **Pandas**: For data processing and manipulation
- **Numpy**: For handling data arrays
- **Matplotlib**: For plotting graphs
- **Scikit-learn**: For data preprocessing (MinMaxScaler)
- **TensorFlow/Keras**: For building and training the LSTM model

## Model Architecture

The model follows an LSTM-based architecture to predict stock prices with the following layers:
- **Input Layer**: Accepts data of shape (daysOfData, 1)
- **LSTM (64 neurons, return_sequences=True)**: learns patterns in the stock price over several days
- **LSTM (64 neurons)**: Learns deeper patterns
- **Dense (32 neurons)**: Learns deeper patterns
- **Dense (1 neuron)**: Outputs a predicted stock price

## Code Walkthrough

### Data Loading and Preprocessing
```python
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
```

- The dataset is downloaded using the **yfinance** library
- Only the **'Close'** prices are used for the prediction
- Data is normalized using **MinMaxScaler()** to improve model performance
- The data is split into **80% training** and **20% testing**

### Create the Model
```python
#create LSTM model
model = Sequential([
    Input(shape = (daysOfData, 1)),
    LSTM(64, return_sequences = True), 
    LSTM(64),
    Dense(32), 
    Dense(1)
])
```

The model is built using the **Sequential API** in TensorFlow/Keras, consisting of LSTM layers to learn from past stock data. The final dense layer outputs the predicted price.

### Compile and Train
```python
#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#create early stopping callback
earlyStopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

#train
model.fit(XTrain, yTrain, epochs = 100, batch_size=32, validation_data=(XTest, yTest), callbacks = [earlyStopping])
```

- The model is compiled with **Adam optimizer** and **mean squared error loss**
- **Early stopping** is used to prevent overfitting and restore the best weights
- Training is conducted for **100 epochs** with a batch size of **32**

### Results and Predictions 
```python
# Predict stock prices using the trained model
predictions = model.predict(XTest)

# Reshape predictions to ensure correct format (column vector)
predictions = predictions.reshape(predictions.shape[0], 1)

# Convert back to original scale
predictions = scaler.inverse_transform(predictions)  

# Plotting the real vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(yTest):], scaler.inverse_transform(np.array(yTest).reshape(-1, 1)), label='Real Prices')
plt.plot(df.index[-len(yTest):], predictions, label='Predicted Prices')
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

Predict stock prices using a trained model and plots both the real and predicted prices over time, comparing them to evaluate the model's performance. The predictions are first reshaped and scaled back to their original values before being plotted with the actual prices.

### Stock Price Prediction Plot
![Prediction Plot](https://github.com/rhb140/Stock-Price-Prediction-LSTM/blob/main/StockPricePredictionImage4.jpg?raw=true)
A plot of real vs predicted stock prices over the test period where blue is the real values based on past stock data and orange is the predicted values.

## Conclusion

The LSTM model predicts Tesla's stock prices, displaying its ability to learn and predict financial trends based on past data. The use of early stopping ensures the model avoids overfitting.

### Author
Created by **Rory Howe-Borges**  
[rhb140](https://github.com/rhb140)

## Citaion
Dataset:
Yahoo Finance. (2025). *Tesla, Inc. (TSLA) Stock Historical Data*. Retrieved from [https://finance.yahoo.com/quote/TSLA/history](https://finance.yahoo.com/quote/TSLA/history)

