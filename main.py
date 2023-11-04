from flask import Flask, render_template, request, redirect, url_for
import func  # Import the functions from func.py
import yfinance as yf
import pandas as pd

app = Flask(__name__)

drawdown = None  # Initialize the drawdown_data variable
image_base64 = None
symbol = None
start_date = None
end_date = None
downloaded_dict = {}

# Route for the main page (form input)
@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        global symbol
        global start_date
        global end_date
        global downloaded

        drawdown = None
        image_base64 = None
        symbol = None
        start_date = None
        end_date = None
        downloaded_dict = {}


        symbol = request.form['symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        data_downloaded = func.download_data(symbol, start_date, end_date)
        downloaded_dict[symbol] = data_downloaded

        if downloaded_dict is not None:
            return render_template('index.html', data_downloaded=data_downloaded)
        else:
            # Handle the case where data_downloaded is not available
            return render_template('index.html')

    # For GET requests, when the form is not submitted
    return render_template('index.html')


@app.route('/describe', methods=['GET', 'POST'])
def describe():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if symbol is not None:
        describe = func.describe(symbol, start_date, end_date,downloaded_dict)
        return render_template('describe.html', describe=describe)
    else:
        return "Describe data is not available."


# Route to calculate drawdown
@app.route('/drawdown', methods=['GET', 'POST'])
def drawdown():
    global symbol
    global start_date
    global end_date
    global drawdown
    global downloaded_dict
    if symbol is not None:
        drawdown = func.drawdown(symbol, start_date, end_date,downloaded_dict)
        return render_template('drawdown_data.html',  drawdown=drawdown)
    else:
        return "Drawdown data is not available."




@app.route('/drawdown_plot', methods=['GET', 'POST'])
def drawdown_plot():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if symbol is not None:
        drawdown_plot = func.drawdown_plot(symbol, start_date, end_date,downloaded_dict)
        return render_template('drawdown_plot.html', drawdown_plot=drawdown_plot)
    else:
        return "Drawdown data is not available. Please calculate drawdown first."



@app.route('/candlestick', methods=['GET', 'POST'])
def candlestick():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if symbol is not None:
        candlestick_json, candlestick_table = func.candlestick_chart(symbol, start_date, end_date,downloaded_dict)
        return render_template('candlestick_plot.html', candlestick_json=candlestick_json, candlestick_table=candlestick_table)
    else:
        return "No plot."



@app.route('/lstm', methods=['GET', 'POST'])
def lstm():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if symbol is not None:
        lstm = func.lstm(symbol, start_date, end_date,downloaded_dict)
        return render_template('lstm.html', lstm=lstm)
    else:
        return "No plot."


# Route to display the drawdown plot
@app.route('/crossover', methods=['GET', 'POST'])
def crossover():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if request.method == 'GET':
        return render_template('crossover.html')
    elif request.method == 'POST':
        if symbol is not None:
            crossover = func.crossover(symbol, start_date, end_date,downloaded_dict)
            return render_template('crossover_result.html', crossover=crossover)
        else:
            return "Crossover data is not available. Please calculate crossover first."


@app.route('/momentum', methods=['GET', 'POST'])
def momentum():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if request.method == 'GET':
        return render_template('momentum.html')
    elif request.method == 'POST':
        if symbol is not None:
            momentum = func.momentum(symbol, start_date, end_date,downloaded_dict)
            return render_template('momentum_result.html', momentum=momentum)
        else:
            return "Momentum data is not available. Please calculate Momentum first."


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if symbol is not None:
        classification = func.classification(symbol, start_date, end_date,downloaded_dict)
        return render_template('logistic_regression.html', classification=classification)
    else:
        return "No sign."


# Route to display the drawdown plot
@app.route('/meanreversion', methods=['GET', 'POST'])
def meanreversion():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if symbol is not None:
        meanreversion = func.meanreversion(symbol, start_date, end_date,downloaded_dict)
        return render_template('mean_reversion.html', meanreversion=meanreversion)
    else:
        return "Mean Reversion data is not available. Please calculate mean reversion first."


@app.route('/trend', methods=['GET', 'POST'])
def trend():
    global symbol
    global start_date
    global end_date
    global downloaded_dict
    if symbol is not None:
        trend = func.trend(symbol, start_date, end_date,downloaded_dict)
        return render_template('trend.html', trend=trend)
    else:
        return "trend data is not available. Please calculate trend first."

if __name__ == '__main__':
    app.run(debug=True)
