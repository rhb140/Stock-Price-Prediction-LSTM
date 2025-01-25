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
![Data Preprocessing Image](https://github.com/rhb140/Stock-Price-Prediction-LSTM/blob/main/StockPricePredictionImage1.jpg?raw=true)

- The dataset is downloaded using the **yfinance** library
- Only the **'Close'** prices are used for the prediction
- Data is normalized using **MinMaxScaler()** to improve model performance
- The data is split into **80% training** and **20% testing**

### Create the Model
![Model Creation Image](https://github.com/rhb140/Stock-Price-Prediction-LSTM/blob/main/StockPricePredictionImage2.jpg?raw=true)

The model is built using the **Sequential API** in TensorFlow/Keras, consisting of LSTM layers to learn from past stock data. The final dense layer outputs the predicted price.

### Compile and Train
![Compile and Train Image](https://github.com/rhb140/Stock-Price-Prediction-LSTM/blob/main/StockPricePredictionImage3.jpg?raw=true)

- The model is compiled with **Adam optimizer** and **mean squared error loss**
- **Early stopping** is used to prevent overfitting and restore the best weights
- Training is conducted for **100 epochs** with a batch size of **32**

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

