import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend instead of interactive GUI
from datetime import datetime, timedelta
import numpy as np
import io
import base64
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask import Flask, render_template, request
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def download_data(symbol, start_date, end_date):
  start_date = datetime.strptime(start_date, '%Y-%m-%d')
  end_date = datetime.strptime(end_date, '%Y-%m-%d')
  data_downloaded = yf.download(symbol, start=start_date, end=end_date)
  return data_downloaded


def describe(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_downloaded = yf.download(symbol, start=start_date, end=end_date)
    db = data_downloaded[['Open','High','Low','Close','Adj Close','Volume']]
    describe = db.describe()
    return describe


def drawdown(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_downloaded = yf.download(symbol, start=start_date, end=end_date)
    df=data_downloaded[['Adj Close']]
    cummax = np.maximum.accumulate(df)
    drawdown = (df - cummax) / cummax
    drawdown = drawdown.describe()
    return drawdown


def drawdown_plot(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_downloaded = yf.download(symbol, start=start_date, end=end_date)
    df=data_downloaded[['Adj Close']]
    cummax = np.maximum.accumulate(df)
    drawdown_p = (df - cummax) / cummax
    dd = drawdown_p.reset_index()
    time_axis = dd['Date']
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, df, label="Investment Value",  linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Investment Value Over Time")
    plt.grid()
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, drawdown_p, label="Drawdown", color='red',linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.title("Drawdown Over Time")
    plt.grid()
    plt.legend()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    drawdown_plot = base64.b64encode(buffer.read()).decode("utf-8")
    return drawdown_plot


def candlestick_chart(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_downloaded = yf.download(symbol, start=start_date, end=end_date)
    # Create a candlestick chart using Plotly
    df = data_downloaded.reset_index()
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'])])
    fig.update_layout(
        title='Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        height=800
    )
    candlestick_json = pio.to_json(fig)
    candlestick_table = df.to_html(classes='table table-bordered', index=False, escape=False)
    return candlestick_json, candlestick_table


def lstm(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_downloaded = yf.download(symbol, start=start_date, end=end_date)
    data = data_downloaded
    data = data[["Close"]]

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values)

    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    n_steps = 7
    n_features = data.shape[1]

    X_train, y_train = [], []
    for i in range(n_steps, len(train_data)):
        X_train.append(train_data[i - n_steps:i])
        y_train.append(train_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test, y_test = [], []
    for i in range(n_steps, len(test_data)):
        X_test.append(test_data[i - n_steps:i])
        y_test.append(test_data[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)

    model = Sequential()
    model.add(LSTM(units=50, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    predictions = model.predict(X_test)

    predictions_unscaled = scaler.inverse_transform(predictions)

    print("LSTM - Test result")
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[train_size + n_steps:], data['Close'].values[train_size + n_steps:], label='Actual')
    plt.plot(data.index[train_size + n_steps:], predictions_unscaled, label='Predicted')
    plt.title(f'LSTM Test Result')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    last_sequence = X_train[-1]

    predicted_values = []

    days = 15
    for i in range(days):
        input_sequence = last_sequence.reshape(1, n_steps, n_features)
        predicted_value = model.predict(input_sequence)[0][0]
        predicted_values.append(predicted_value)
        last_sequence = np.append(last_sequence[1:], predicted_value)

    predicted_values_unscaled = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
    predicted_values_unscaled = predicted_values_unscaled.tolist()
    today = datetime.today()

    next_days = []
    for i in range(days):
        day = today + timedelta(days=i)
        day = day.strftime('%Y-%m-%d')
        next_days.append(day[5:])

    print(f"Predicted Prices for Next {days} Days")
    plt.figure(figsize=(12, 6))
    plt.plot(next_days, predicted_values_unscaled, label='Predicted')
    plt.title(f'Price Prediction for Next {days} Days')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    lstm = base64.b64encode(buffer.read()).decode("utf-8")
    return lstm





def crossover(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data_downloaded = yf.download(symbol, start=start_date, end=end_date)
    short_period = int(request.form['short_period'])
    long_period = int(request.form['long_period'])
    data = data_downloaded
    data['EMA_short'] = data['Close'].ewm(span=short_period, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_period, adjust=False).mean()
    data['Signal'] = 0
    data.loc[data['EMA_short'] > data['EMA_long'], 'Signal'] = 1
    data.loc[data['EMA_short'] < data['EMA_long'], 'Signal'] = -1
    data['Return'] = data['Close'].pct_change() * data['Signal'].shift(1)
    data['Cumulative_Return'] = (1 + data['Return']).cumprod()
    dd =data.reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(data['Cumulative_Return'])
    plt.title('EMA Crossover Strategy')
    plt.xlabel(dd['Date'])
    plt.ylabel('Cumulative Return')
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    plot_crossover = base64.b64encode(buffer.read()).decode("utf-8")
    return plot_crossover


def momentum(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    lookback_period = int(request.form['lookback_period'])
    if 'Close' not in data.columns:
        raise ValueError("The 'Close' column is required in the input DataFrame.")
    data['Return'] = data['Close'].pct_change()
    data['Momentum'] = data['Return'].rolling(lookback_period).sum()
    data['Signal'] = 0  # 0 represents no action
    data.loc[data['Momentum'] > 0, 'Signal'] = 1  # Buy signal for positive momentum
    data.loc[data['Momentum'] < 0, 'Signal'] = -1  # Sell signal for negative momentum

    data['Strategy_Return'] = data['Return'] * data['Signal'].shift(1)

    data['Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()
    dd =data.reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(data['Cumulative_Return'])
    plt.title('Momentum Strategy')
    plt.xlabel(dd['Date'])
    plt.ylabel('Cumulative Return')
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    momentum = base64.b64encode(buffer.read()).decode("utf-8")
    return momentum


def classification(symbol, start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Pct'] = data['Close'].pct_change()
    data['Pct_Sign'] = np.where(data['Pct'] > 0, 1, 0)

    data['Open_pct'] = data['Open'].pct_change()
    data['Open_Sign'] = np.where(data['Open_pct'] > 0, 1, 0)

    data['Volume_pct'] = data['Volume'].pct_change()
    data['Volume_Sign'] = np.where(data['Volume_pct'] > 0, 1, 0)

    data['Tomorrow_Return'] = data['Pct_Sign'].shift(-1)
    last_row = data.iloc[-1]

    data = data.dropna()

    X = data[['Pct_Sign','Open_Sign', 'Volume_Sign']]  # Use 'Pct' as the feature
    y = data['Tomorrow_Return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    last_features = last_row[['Pct_Sign','Open_Sign', 'Volume_Sign']]

    predicted_tomorrow_sign = model.predict(last_features.values.reshape(1, -1))

    if predicted_tomorrow_sign>0:
        tomorrow_is = 'Positive';
    else:
        tomorrow_is = 'Negative';

    df_classification = pd.DataFrame({'Prediction for Tomorrow': [tomorrow_is], 'Accuracy': [accuracy]})

    return df_classification



def mean_reversion(symbol, start_date, end_date, lookback_period=20, z_score_threshold=2):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    data['RollingMean'] = data['Close'].rolling(window=lookback_period).mean()
    data['RollingStd'] = data['Close'].rolling(window=lookback_period).std()

    data['ZScore'] = (data['Close'] - data['RollingMean']) / data['RollingStd']

    data['Signal'] = 0

    data.loc[data['ZScore'] < -z_score_threshold, 'Signal'] = 1
    data.loc[data['ZScore'] > z_score_threshold, 'Signal'] = -1

    data['Return'] = data['Close'].pct_change() * data['Signal'].shift(1)

    data['CumulativeReturn'] = (1 + data['Return']).cumprod()

    dd =data.reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(data['CumulativeReturn'])
    plt.title('Mean Reversion Strategy')
    plt.xlabel(dd['Date'])
    plt.ylabel('Cumulative Return')
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    plot_reversion = base64.b64encode(buffer.read()).decode("utf-8")
    return plot_reversion


def trend(symbol, start_date, end_date, short_window = 10, long_window = 50):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

    data['Signal'] = 0

    data.loc[data['Short_MA'] > data['Long_MA'], 'Signal'] = 1  # Golden Cross (short MA crosses above long MA)
    data.loc[data['Short_MA'] < data['Long_MA'], 'Signal'] = -1  # Death Cross (short MA crosses below long MA)

    data['Return'] = data['Signal'].shift(1) * data['Close'].pct_change()

    data['CumulativeReturn'] = (1 + data['Return']).cumprod()

    data['Peak'] = data['CumulativeReturn'].cummax()
    data['Drawdown'] = (data['Peak'] - data['CumulativeReturn']) / data['Peak']

    dd =data.reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(dd['Date'], data['CumulativeReturn'], label='Cumulative Return')
    #plt.plot(dd['Date'], data['Drawdown'], label='Drawdown', color='red')
    plt.title('Trend-Following Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)
    plot_trend = base64.b64encode(buffer.read()).decode("utf-8")
    return plot_trend