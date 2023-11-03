from flask import Flask, render_template, request, redirect, url_for
import func  # Import the functions from func.py
import yfinance as yf
import pandas as pd

app = Flask(__name__)

drawdown_data = None  # Initialize the drawdown_data variable
image_base64 = None
drawdown_desc = None
symbol = None
start_date = None
end_date = None


# Route for the main page (form input)
@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        global symbol
        global start_date
        global end_date

        symbol = request.form['symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        data_downloaded = func.download_data(symbol, start_date, end_date)

        if data_downloaded is not None:
            return render_template('index.html', data_downloaded=data_downloaded)
        else:
            # Handle the case where data_downloaded is not available
            return render_template('index.html')

    # For GET requests, when the form is not submitted
    return render_template('index.html')



# Route to calculate drawdown
@app.route('/describe', methods=['GET', 'POST'])
def calculate_describe():
    global symbol
    global start_date
    global end_date

    if symbol is not None:
        data_describe = func.data_describe(symbol, start_date, end_date)
        return render_template('describe.html', data_describe=data_describe)
    else:
        return render_template('index.html', error_message="Data not available")




# Route to calculate drawdown
@app.route('/drawdown', methods=['GET', 'POST'])
def drawdown():
    global drawdown_data
    global drawdown_desc
    if symbol is not None:
        drawdown_desc = func.drawdown(symbol, start_date, end_date)
        return render_template('index.html',  drawdown_desc=drawdown_desc)
    else:
        return "Drawdown data is not available."




# Route to display the drawdown plot
@app.route('/ddplot', methods=['GET', 'POST'])
def drawdown_plot():
    if symbol is not None:
        image_base64 = func.drawdown_plot(symbol, start_date, end_date)
        return render_template('index.html', image_base64=image_base64)
    else:
        return "Drawdown data is not available. Please calculate drawdown first."



@app.route('/candlestick', methods=['GET', 'POST'])
def candlestick():
    if symbol is not None:
        candlestick_json, candlestick_table = func.candlestick_chart(symbol, start_date, end_date)
        return render_template('index.html', candlestick_json=candlestick_json, candlestick_table=candlestick_table)
    else:
        return "No plot."



@app.route('/lstm', methods=['GET', 'POST'])
def lstm():
    if symbol is not None:
        lstm = func.lstm(symbol, start_date, end_date)
        return render_template('index.html', lstm=lstm)
    else:
        return "No plot."


# Route to display the drawdown plot
@app.route('/crossover', methods=['GET', 'POST'])
def crossover():
    if request.method == 'GET':
        return render_template('crossover.html')
    elif request.method == 'POST':
        if symbol is not None:
            plot_crossover = func.crossover(symbol, start_date, end_date)
            return render_template('index.html', plot_crossover=plot_crossover)
        else:
            return "Crossover data is not available. Please calculate crossover first."


@app.route('/momentum', methods=['GET', 'POST'])
def momentum():
    if request.method == 'GET':
        return render_template('momentum.html')
    elif request.method == 'POST':
        if symbol is not None:
            momentum = func.momentum(symbol, start_date, end_date)
            return render_template('index.html', momentum=momentum)
        else:
            return "Momentum data is not available. Please calculate Momentum first."


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    if symbol is not None:
        df_classification = func.classification(symbol, start_date, end_date)
        return render_template('index.html', df_classification=df_classification)
    else:
        return "No sign."


# Route to display the drawdown plot
@app.route('/meanreversion', methods=['GET', 'POST'])
def mean_reversion():
    if symbol is not None:
        mean_reversion = func.mean_reversion(symbol, start_date, end_date)
        return render_template('index.html', mean_reversion=mean_reversion)
    else:
        return "Mean Reversion data is not available. Please calculate mean reversion first."


@app.route('/trend', methods=['GET', 'POST'])
def trend():
    if symbol is not None:
        trend = func.trend(symbol, start_date, end_date)
        return render_template('index.html', trend=trend)
    else:
        return "trend data is not available. Please calculate trend first."

if __name__ == '__main__':
    app.run(debug=True)
