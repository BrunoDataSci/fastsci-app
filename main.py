from flask import Flask, render_template, request, redirect, url_for, session
import func  # Import the functions from func.py
import yfinance as yf
import pandas as pd

app = Flask(__name__)
app.secret_key = '123456'  # Set a secret key for the session


data_downloaded = None  # Initialize the data_downloaded variable
drawdown_data = None  # Initialize the drawdown_data variable
image_base64 = None
drawdown_desc = None
symbol = None

# Route for the main page (form input)
@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        symbol = request.form['symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Download data and store it in the session
        data_downloaded = func.download_data(symbol, start_date, end_date)
        session['data_downloaded'] = data_downloaded

        # Check if data_downloaded is available and not None
        if data_downloaded is not None and not data_downloaded.empty:
            return render_template('index.html', data_downloaded=data_downloaded)
        else:
            # Handle the case where data_downloaded is not available
            return render_template('index.html', error_message="Data not available.")

    return render_template('index.html')



@app.route('/describe', methods=['GET', 'POST'])
def calculate_describe():
    data_downloaded = session.get('data_downloaded')
    if data_downloaded is not None:
        describe = func.data_describe(data_downloaded)
        return render_template('data_downloaded.html', describe=describe)
    else:
        return "Data not available."



# Route to calculate drawdown
@app.route('/drawdown', methods=['GET', 'POST'])
def drawdown():
    data_downloaded = session.get('data_downloaded')
    global drawdown_data
    global drawdown_desc
    if data_downloaded is not None:
        drawdown_desc = func.drawdown(data_downloaded)
        return render_template('drawdown_desc.html',  drawdown_desc=drawdown_desc)
    else:
        return "Drawdown data is not available."




# Route to display the drawdown plot
@app.route('/ddplot', methods=['GET', 'POST'])
def drawdown_plot():
    data_downloaded = session.get('data_downloaded')
    if data_downloaded is not None:
        image_base64 = func.drawdown_plot(data_downloaded)
        return render_template('drawdown_plot.html', image_base64=image_base64)
    else:
        return "Drawdown data is not available. Please calculate drawdown first."



@app.route('/candlestick', methods=['GET', 'POST'])
def candlestick():
    data_downloaded = session.get('data_downloaded')
    if data_downloaded is not None:
        candlestick_json, candlestick_table = func.candlestick_chart(data_downloaded)
        return render_template('candlestick_plot.html', candlestick_json=candlestick_json, candlestick_table=candlestick_table)
    else:
        return "No plot."



@app.route('/lstm', methods=['GET', 'POST'])
def lstm():
    data_downloaded = session.get('data_downloaded')
    if data_downloaded is not None:
        lstm = func.lstm(data_downloaded)
        return render_template('lstm.html', lstm=lstm)
    else:
        return "No plot."


# Route to display the drawdown plot
@app.route('/crossover', methods=['GET', 'POST'])
def crossover():
    data_downloaded = session.get('data_downloaded')
    if request.method == 'GET':
        return render_template('crossover.html')
    elif request.method == 'POST':
        if data_downloaded is not None:
            plot_crossover = func.crossover(data_downloaded)
            return render_template('crossover_result.html', plot_crossover=plot_crossover)
        else:
            return "Crossover data is not available. Please calculate crossover first."


@app.route('/momentum', methods=['GET', 'POST'])
def momentum():
    data_downloaded = session.get('data_downloaded')
    if request.method == 'GET':
        return render_template('momentum.html')
    elif request.method == 'POST':
        if data_downloaded is not None:
            momentum = func.momentum(data_downloaded)
            return render_template('momentum_result.html', momentum=momentum)
        else:
            return "Momentum data is not available. Please calculate Momentum first."


@app.route('/classification', methods=['GET', 'POST'])
def classification():
    data_downloaded = session.get('data_downloaded')
    if data_downloaded is not None:
        df_classification = func.classification(data_downloaded)
        return render_template('logistic_regression.html', df_classification=df_classification)
    else:
        return "No sign."


# Route to display the drawdown plot
@app.route('/meanreversion', methods=['GET', 'POST'])
def mean_reversion():
    data_downloaded = session.get('data_downloaded')
    if data_downloaded is not None:
        mean_reversion = func.mean_reversion(data_downloaded)
        return render_template('mean_reversion.html', mean_reversion=mean_reversion)
    else:
        return "Mean Reversion data is not available. Please calculate mean reversion first."


@app.route('/trend', methods=['GET', 'POST'])
def trend():
    data_downloaded = session.get('data_downloaded')
    if data_downloaded is not None:
        trend = func.trend(data_downloaded)
        return render_template('trend.html', trend=trend)
    else:
        return "trend data is not available. Please calculate trend first."

if __name__ == '__main__':
    app.run(debug=True)
