import dash
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from ta.volatility import AverageTrueRange  # ATR indicator

# Set external stylesheets for better aesthetics
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def load_data(stock_symbols):
    data_frames = []
    for stock_symbol in stock_symbols:
        if stock_symbol == 'ASELS':
            data_path = "ASELS.csv"
        elif stock_symbol == 'THYAO':
            data_path = "THYAO.csv"
        elif stock_symbol == 'AFYON':
            data_path = "AFYON.csv"
        elif stock_symbol == 'AEFES':
            data_path = "AEFES.csv"
        else:
            raise ValueError(f"Unknown stock symbol: {stock_symbol}")

        data = pd.read_csv(data_path)
        data['TRADE DATE'] = pd.to_datetime(data['TRADE DATE'])
        data['Stock'] = stock_symbol  # Add stock symbol to identify the stock
        data_frames.append(data)

    combined_data = pd.concat(data_frames)
    return combined_data


def load_models_for_stocks(stock_symbols):
    models = {}
    for stock_symbol in stock_symbols:
        if stock_symbol == 'ASELS':
            model_path ="best_model ASELS.keras"
        elif stock_symbol == 'THYAO':
            model_path = "best_model THYAO.keras"
        elif stock_symbol == 'AFYON':
            model_path = "best_model AFYON.keras"
        elif stock_symbol == 'AEFES':
            model_path = "best_model AEFES.keras"
        else:
            raise ValueError(f"Unknown stock symbol: {stock_symbol}")

        model = load_model(model_path)
        models[stock_symbol] = model
    return models


def feature_engineering(data):
    # Feature engineering steps
    data['TRADE DATE'] = pd.to_datetime(data['TRADE DATE'])

    # Calculate the daily closing opening difference
    data['CLOSING_OPEN_DIFF'] = data['CLOSING PRICE'] - data['OPENING PRICE']

    # Calculate the 7-14-21 day moving average
    data['7_DAY_MOVING_AVG'] = data.groupby('Stock')['CLOSING PRICE'].transform(lambda x: x.rolling(window=7).mean())
    data['14_DAY_MOVING_AVG'] = data.groupby('Stock')['CLOSING PRICE'].transform(lambda x: x.rolling(window=14).mean())
    data['21_DAY_MOVING_AVG'] = data.groupby('Stock')['CLOSING PRICE'].transform(lambda x: x.rolling(window=21).mean())

    # Provide the model with historical trading activity.
    for lag in range(1, 8):
        data[f'TOTAL_TRADED_VOLUME_LAG_{lag}'] = data.groupby('Stock')['TOTAL TRADED VOLUME'].shift(lag - 1)
        data[f'TOTAL_TRADED_VALUE_LAG_{lag}'] = data.groupby('Stock')['TOTAL TRADED VALUE'].shift(lag - 1)

    # Capture short-term trends and patterns.
    for lag in range(1, 8):
        data[f'CLOSING_PRICE_LAG_{lag}'] = data.groupby('Stock')['CLOSING PRICE'].shift(lag - 1)

    # Features of the previous day
    data["PREVIOUS_DAY_OPENING_DIFF"] = data.groupby('Stock')["OPENING PRICE"].shift(1) - data["OPENING PRICE"]

    data["7_DAY_MOVING_VWAP_AVG"] = data.groupby('Stock')["VWAP"].transform(lambda x: x.rolling(window=7).mean())

    # Calculate the max price difference in one day
    data['PRICE_RANGE'] = data['HIGHEST PRICE'] - data['LOWEST PRICE']

    # EMAs give more weight to recent prices, potentially capturing more recent trends.
    data['7_DAY_EMA'] = data.groupby('Stock')['CLOSING PRICE'].transform(lambda x: x.ewm(span=7, adjust=False).mean())

    # Calculate ATR
    def calc_atr(group):
        atr_indicator = AverageTrueRange(
            high=group['HIGHEST PRICE'],
            low=group['LOWEST PRICE'],
            close=group['CLOSING PRICE'],
            window=14
        )
        group['ATR'] = atr_indicator.average_true_range()
        return group

    data = data.groupby('Stock').apply(calc_atr)
    data.reset_index(drop=True, inplace=True)

    data = data.dropna()
    return data


def create_sequences(data, target, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i + window_size])
        labels.append(target[i + window_size])
    return np.array(sequences), np.array(labels)


app.layout = html.Div([
    html.H1("Autonomous Stock Trading Simulation", style={'textAlign': 'center'}),

    # Disclaimer here
    html.Div([
        html.P(
            "This simulation is for educational purposes only and does not constitute financial advice. "
            "For actual investment decisions, consult a professional advisor.",
            style={
                'color': 'red',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'fontSize': '16px',
            }
        ),
    ], style={'padding': '10px 0'}),

    html.Div([
        html.Label("Select Stocks:"),
        dcc.Dropdown(
            id='stock_selector',
            options=[
                {'label': 'ASELS', 'value': 'ASELS'},
                {'label': 'THYAO', 'value': 'THYAO'},
                {'label': 'AFYON', 'value': 'AFYON'},
                {'label': 'AEFES', 'value': 'AEFES'},
            ],
            value=['ASELS', 'THYAO'],  # Default value
            multi=True,
            style={'width': '50%'}
        ),
        html.Br(),

        html.Label("Initial Balance (TL):"),
        dcc.Input(id="initial_balance", type="number", value=100000, min=0, step=1000),
        html.Br(),
        html.Label("Transaction Cost (As a ratio, e.g. 0.002 = 0.2%):"),
        dcc.Input(id="transaction_cost", type="number", value=0.002, min=0, step=0.0001),
        html.Br(),
        html.Label("Position Size Percent (As a %, e.g. 10 = %10):"),
        dcc.Input(id="position_size_percent", type="number", value=10, min=0, max=100, step=1),
        html.Br(),
        html.Label("Stop-Loss Percent (As a %, e.g. 5 = %5):"),
        dcc.Input(id="stop_loss_percent", type="number", value=5, min=0, max=100, step=0.1),
        html.Br(),
        html.Label("Take-Profit Percent (As a %, e.g. 10 = %10):"),
        dcc.Input(id="take_profit_percent", type="number", value=10, min=0, max=100, step=0.1),
        html.Br(),
        html.Button("Start Simulation", id="start_simulation", n_clicks=0, style={'marginTop': '10px'})
    ], style={'padding': '20px', 'border': '1px solid #d9d9d9', 'borderRadius': '5px'}),

    # Store simulation results to avoid redundant computations
    dcc.Store(id='simulation_results'),

    dcc.Tabs(id="tabs", value='balance', children=[
        dcc.Tab(label="Balance Chart", value="balance"),
        dcc.Tab(label="Portfolio Composition", value="portfolio"),
        dcc.Tab(label="Estimated and Actual Price Comparison", value="price_comparison")
    ], style={'marginTop': '20px'}),

    html.Div(id="tabs_content", style={'padding': '20px'}),

    html.Div(id="transaction_summary", style={'marginTop': '20px'}),

    # Additional disclaimer at the bottom
    html.Div([
        html.P("Warning: This simulation does not represent real-time data and transactions."
               " Carefully evaluate all risks that may affect your investment decisions.",
               style={
                   'color': 'gray',
                   'fontStyle': 'italic',
                   'textAlign': 'center',
                   'fontSize': '12px',
                   'paddingTop': '20px',
               }
        ),
    ]),
])


def run_simulation(data, initial_balance, transaction_cost, position_size_percent, stop_loss_percent, take_profit_percent):
    balance = initial_balance
    positions = {}  # Dictionary to hold the number of shares owned for each stock
    balance_history = []
    transactions = []
    buy_dates, buy_prices, buy_stocks = [], [], []
    sell_dates, sell_prices, sell_stocks = [], [], []
    positions_info = {}

    earnings_per_stock = {}
    total_shares_per_stock = {}

    data = data.sort_values(by='TRADE DATE').reset_index(drop=True)
    dates = data['TRADE DATE'].unique()

    for current_date in dates:
        daily_data = data[data['TRADE DATE'] == current_date]

        for stock_symbol in daily_data['Stock'].unique():
            stock_data = daily_data[daily_data['Stock'] == stock_symbol].iloc[0]

            current_price = stock_data['CLOSING PRICE']
            predicted_price = stock_data['Predicted_Price']
            atr = stock_data['ATR']

            # Ensure current_price and atr are positive
            if current_price <= 0 or atr <= 0 or np.isnan(atr):
                continue

            # Set dynamic thresholds based on ATR
            buy_threshold = current_price + (atr * 0.5)
            sell_threshold = current_price - (atr * 0.5)

            # Calculate position size
            position_value = balance * (position_size_percent / 100)

            shares_owned = positions.get(stock_symbol, 0)
            position_info = positions_info.get(stock_symbol, {'entry_price': None, 'stop_loss_price': None, 'take_profit_price': None, 'total_cost': None})

            # Trading logic
            if shares_owned == 0:
                # Check buy conditions
                if predicted_price > buy_threshold and balance >= current_price:
                    shares_to_buy = int(position_value // (current_price * (1 + transaction_cost)))
                    if shares_to_buy > 0:
                        total_cost = shares_to_buy * current_price * (1 + transaction_cost)
                        balance -= total_cost
                        positions[stock_symbol] = shares_to_buy
                        transactions.append(f"{current_date.date()} Purchase {stock_symbol}: {shares_to_buy} shares @ {current_price:.2f} TL/share, Total Cost: {total_cost:,.2f} TL")
                        buy_dates.append(current_date)
                        buy_prices.append(current_price)
                        buy_stocks.append(stock_symbol)
                        # Set entry price and stop-loss/take-profit levels
                        entry_price = current_price
                        stop_loss_price = entry_price * (1 - stop_loss_percent / 100)
                        take_profit_price = entry_price * (1 + take_profit_percent / 100)
                        positions_info[stock_symbol] = {
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss_price,
                            'take_profit_price': take_profit_price,
                            'total_cost': total_cost
                        }
                        total_shares_per_stock[stock_symbol] = total_shares_per_stock.get(stock_symbol, 0) + shares_to_buy
            else:
                # Position info
                entry_price = position_info['entry_price']
                stop_loss_price = position_info['stop_loss_price']
                take_profit_price = position_info['take_profit_price']
                total_cost = position_info['total_cost']

                # Check stop-loss condition
                if current_price <= stop_loss_price:
                    # Sell due to stop-loss
                    total_revenue = shares_owned * current_price * (1 - transaction_cost)
                    balance += total_revenue
                    profit = total_revenue - total_cost
                    earnings_per_stock[stock_symbol] = earnings_per_stock.get(stock_symbol, 0) + profit
                    transactions.append(f"{current_date.date()} Stop-Loss Sell {stock_symbol}: {shares_owned} shares @ {current_price:.2f} TL/share, Total Revenue: {total_revenue:,.2f} TL")
                    sell_dates.append(current_date)
                    sell_prices.append(current_price)
                    sell_stocks.append(stock_symbol)
                    positions[stock_symbol] = 0
                    positions_info.pop(stock_symbol, None)
                # Check take-profit condition
                elif current_price >= take_profit_price:
                    # Sell due to take-profit
                    total_revenue = shares_owned * current_price * (1 - transaction_cost)
                    balance += total_revenue
                    profit = total_revenue - total_cost
                    earnings_per_stock[stock_symbol] = earnings_per_stock.get(stock_symbol, 0) + profit
                    transactions.append(f"{current_date.date()} Take-Profit Sell {stock_symbol}: {shares_owned} shares @ {current_price:.2f} TL/share, Total Revenue: {total_revenue:,.2f} TL")
                    sell_dates.append(current_date)
                    sell_prices.append(current_price)
                    sell_stocks.append(stock_symbol)
                    positions[stock_symbol] = 0
                    positions_info.pop(stock_symbol, None)
                # Check sell condition based on prediction
                elif predicted_price < sell_threshold:
                    total_revenue = shares_owned * current_price * (1 - transaction_cost)
                    balance += total_revenue
                    profit = total_revenue - total_cost
                    earnings_per_stock[stock_symbol] = earnings_per_stock.get(stock_symbol, 0) + profit
                    transactions.append(f"{current_date.date()} Sell (By Estimate) {stock_symbol}: {shares_owned} shares @ {current_price:.2f} TL/share, Total Revenue: {total_revenue:,.2f} TL")
                    sell_dates.append(current_date)
                    sell_prices.append(current_price)
                    sell_stocks.append(stock_symbol)
                    positions[stock_symbol] = 0
                    positions_info.pop(stock_symbol, None)

        # Update total portfolio value
        portfolio_value = balance
        for stock, shares in positions.items():
            if shares > 0:
                current_stock_price = daily_data[daily_data['Stock'] == stock]['CLOSING PRICE'].iloc[0]
                portfolio_value += shares * current_stock_price

        balance_history.append(portfolio_value)

    # Calculate final balance including the value of owned shares
    final_balance = portfolio_value

    # Calculate performance metrics
    # Daily Returns
    balance_array = np.array(balance_history)
    daily_returns = balance_array[1:] / balance_array[:-1] - 1
    mean_daily_return = np.mean(daily_returns)
    std_daily_return = np.std(daily_returns)
    risk_free_rate = 0.0  # Assuming zero risk-free rate

    # Sharpe Ratio
    if std_daily_return != 0:
        sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return * np.sqrt(252)  # Annualized Sharpe Ratio
    else:
        sharpe_ratio = 0.0

    # Maximum Drawdown
    running_max = np.maximum.accumulate(balance_array)
    drawdowns = (balance_array - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Prepare results
    simulation_results = {
        'balance_history': balance_history,
        'transactions': transactions,
        'final_balance': final_balance,
        'final_cash': balance,
        'positions': positions,
        'buy_dates': [str(date) for date in buy_dates],
        'buy_prices': buy_prices,
        'buy_stocks': buy_stocks,
        'sell_dates': [str(date) for date in sell_dates],
        'sell_prices': sell_prices,
        'sell_stocks': sell_stocks,
        'initial_balance': initial_balance,
        'mean_daily_return': mean_daily_return,
        'std_daily_return': std_daily_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'max_profit': max(balance_history) - initial_balance,
        'earnings_per_stock': earnings_per_stock,
        'total_shares_per_stock': total_shares_per_stock,
    }

    return simulation_results


@app.callback(
    Output('simulation_results', 'data'),
    [Input('start_simulation', 'n_clicks')],
    [State('stock_selector', 'value'),
     State("initial_balance", "value"),
     State("transaction_cost", "value"),
     State("position_size_percent", "value"),
     State("stop_loss_percent", "value"),
     State("take_profit_percent", "value")]
)
def run_and_store_simulation(n_clicks, stock_symbols, initial_balance, transaction_cost, position_size_percent, stop_loss_percent, take_profit_percent):
    if n_clicks == 0:
        return None

    # Validate inputs
    try:
        initial_balance = float(initial_balance)
        transaction_cost = float(transaction_cost)
        position_size_percent = float(position_size_percent)
        stop_loss_percent = float(stop_loss_percent)
        take_profit_percent = float(take_profit_percent)

        # Input validations
        if initial_balance <= 0:
            raise ValueError("Initial balance must be greater than zero.")
        if not (0 <= transaction_cost <= 1):
            raise ValueError("Transaction cost must be between 0 and 1.")
        if not (0 < position_size_percent <= 100):
            raise ValueError("Position size percent must be between 0 and 100.")
        if not (0 < stop_loss_percent < 100):
            raise ValueError("Stop-loss percent must be between 0 and 100.")
        if not (0 < take_profit_percent < 100):
            raise ValueError("Take-profit percent must be between 0 and 100.")
    except Exception as e:
        return {'error': str(e)}

    # Load data and models based on selected stocks
    try:
        data = load_data(stock_symbols)
        data = feature_engineering(data)
        models = load_models_for_stocks(stock_symbols)
    except Exception as e:
        return {'error': str(e)}

    data_with_predictions_list = []

    for stock_symbol in stock_symbols:
        stock_data = data[data['Stock'] == stock_symbol].copy()

        # Select the important features
        important_features = ['TRADE DATE', 'CLOSING PRICE', "ATR", 'CLOSING_OPEN_DIFF', "TRADED VOLUME AT OPENING SESSION",
                              "REMAINING ASK",
                              '7_DAY_MOVING_AVG', '14_DAY_MOVING_AVG', '21_DAY_MOVING_AVG',
                              "CHANGE TO PREVIOUS CLOSING (%)",
                              'TOTAL TRADED VALUE', 'TOTAL TRADED VOLUME', 'OPENING PRICE', "OPENING SESSION PRICE",
                              "7_DAY_EMA",
                              'SUSPENDED', 'TRADED VALUE AT CLOSING SESSION', "NUMBER OF CONTRACTS AT CLOSING SESSION",
                              'CLOSING_PRICE_LAG_1', 'CLOSING_PRICE_LAG_2', 'CLOSING_PRICE_LAG_3', 'CLOSING_PRICE_LAG_4',
                              'CLOSING_PRICE_LAG_5', 'CLOSING_PRICE_LAG_6', 'CLOSING_PRICE_LAG_7',
                              'PRICE_RANGE', "LOWEST PRICE", "VWAP", "TRADED VALUE AT OPENING SESSION",
                              "TOTAL_TRADED_VALUE_LAG_1", 'TOTAL_TRADED_VALUE_LAG_2', "TOTAL_TRADED_VALUE_LAG_3",
                              "TOTAL_TRADED_VALUE_LAG_4", 'TOTAL_TRADED_VALUE_LAG_5', "TOTAL_TRADED_VALUE_LAG_6",
                              "TOTAL_TRADED_VALUE_LAG_7",
                              'TOTAL_TRADED_VOLUME_LAG_1', 'TOTAL_TRADED_VOLUME_LAG_2', 'TOTAL_TRADED_VOLUME_LAG_3',
                              'TOTAL_TRADED_VOLUME_LAG_4', 'TOTAL_TRADED_VOLUME_LAG_5', 'TOTAL_TRADED_VOLUME_LAG_6',
                              'TOTAL_TRADED_VOLUME_LAG_7',
                              "HIGHEST PRICE", "PREVIOUS_DAY_OPENING_DIFF", "7_DAY_MOVING_VWAP_AVG"]

        # Ensure that all important features are present in the dataset
        missing_features = set(important_features) - set(stock_data.columns)
        if missing_features:
            return {'error': f"The following features are missing in the data for stock {stock_symbol}: {missing_features}"}

        selected_data = stock_data[important_features]

        # Ensure no missing values
        selected_data = selected_data.dropna()

        # Split the dataset
        train_size = int(len(selected_data) * 0.8)
        train_data = selected_data[:train_size]
        test_data = selected_data[train_size:]

        # Separate the features and target variable
        feature_columns = important_features[3:]  # Exclude the date and closing price columns
        train_features = train_data[feature_columns]
        train_target = train_data['CLOSING PRICE']

        test_features = test_data[feature_columns]
        test_target = test_data['CLOSING PRICE']

        # Scale the data
        scaler = MinMaxScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # Create sequences
        window_size = 7  # We will predict the next day using 7 days of data
        x_test, y_test = create_sequences(test_features, test_target.values, window_size)

        # Generate predictions
        model = models[stock_symbol]
        predictions = model.predict(x_test)

        data_with_predictions = test_data.iloc[window_size:].copy()
        data_with_predictions['Predicted_Price'] = predictions.ravel()
        data_with_predictions = data_with_predictions.reset_index(drop=True)
        data_with_predictions['Stock'] = stock_symbol
        data_with_predictions_list.append(data_with_predictions)

    # Combine data from all stocks
    data_with_predictions_combined = pd.concat(data_with_predictions_list).sort_values(['TRADE DATE', 'Stock']).reset_index(drop=True)

    # Run the trading simulation
    simulation_results = run_simulation(
        data_with_predictions_combined,
        initial_balance,
        transaction_cost,
        position_size_percent,
        stop_loss_percent,
        take_profit_percent
    )

    # Include data needed for plotting in simulation_results
    simulation_results['dates'] = data_with_predictions_combined['TRADE DATE'].astype(str).tolist()
    simulation_results['data_with_predictions'] = data_with_predictions_combined.to_dict('records')
    simulation_results['stock_symbols'] = stock_symbols  # Include stock symbols

    return simulation_results


@app.callback(
    Output("tabs_content", "children"),
    [Input("tabs", "value"),
     Input('simulation_results', 'data')]
)
def render_content(tab, simulation_results):
    if simulation_results is None:
        return html.Div("To start the simulation, enter the parameters and click the 'Start Simulation' button.")

    if 'error' in simulation_results:
        return html.Div(f"Error: {simulation_results['error']}", style={'color': 'red', 'fontWeight': 'bold'})

    # Extract data from simulation_results
    balance_history = simulation_results['balance_history']
    dates = simulation_results['dates']
    stock_symbols = simulation_results.get('stock_symbols', [])
    data_with_predictions = pd.DataFrame(simulation_results['data_with_predictions'])

    if tab == "balance":
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=dates, y=balance_history, mode='lines', name='Balance'))

        # Update layout for interactivity
        fig.update_layout(
            title=f"Daily Portfolio Value",
            xaxis_title="Date",
            yaxis_title="Balance (TL)",
            hovermode='x unified'
        )

        return dcc.Graph(figure=fig)

    elif tab == "portfolio":
        positions = simulation_results['positions']
        # Filter out stocks with zero shares
        positions = {stock: shares for stock, shares in positions.items() if shares > 0}

        if positions:
            labels = list(positions.keys())
            values = list(positions.values())
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig.update_layout(title='Current Portfolio Composition')
            return dcc.Graph(figure=fig)
        else:
            return html.Div("No holdings in the portfolio currently.")

    elif tab == "price_comparison":
        # Display separate graphs per stock
        content = []
        for stock_symbol in stock_symbols:
            stock_data = data_with_predictions[data_with_predictions['Stock'] == stock_symbol]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=stock_data['TRADE DATE'],
                y=stock_data['CLOSING PRICE'],
                mode='lines',
                name=f'Actual Price {stock_symbol}',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=stock_data['TRADE DATE'],
                y=stock_data['Predicted_Price'],
                mode='lines',
                name=f'Predicted Price {stock_symbol}',
                line=dict(color='orange')
            ))

            # Plot buy and sell points for this stock
            buy_data = pd.DataFrame({
                'Date': simulation_results['buy_dates'],
                'Price': simulation_results['buy_prices'],
                'Stock': simulation_results['buy_stocks']
            })
            buy_data = buy_data[buy_data['Stock'] == stock_symbol]

            sell_data = pd.DataFrame({
                'Date': simulation_results['sell_dates'],
                'Price': simulation_results['sell_prices'],
                'Stock': simulation_results['sell_stocks']
            })
            sell_data = sell_data[sell_data['Stock'] == stock_symbol]

            fig.add_trace(go.Scatter(
                x=buy_data['Date'],
                y=buy_data['Price'],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Purchase'
            ))
            fig.add_trace(go.Scatter(
                x=sell_data['Date'],
                y=sell_data['Price'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Sell'
            ))

            # Update layout for interactivity
            fig.update_layout(
                title=f"Estimated and Actual Price Comparison for {stock_symbol}",
                xaxis_title="Date",
                yaxis_title="Price (TL)",
                hovermode='x unified'
            )

            content.append(html.Div([
                html.H3(f"{stock_symbol} Price Comparison"),
                dcc.Graph(figure=fig)
            ], style={'marginBottom': '40px'}))

        return html.Div(content)


@app.callback(
    Output("transaction_summary", "children"),
    [Input('simulation_results', 'data')]
)
def update_transaction_summary(simulation_results):
    if simulation_results is None:
        return ""

    if 'error' in simulation_results:
        return html.Div(f"Error: {simulation_results['error']}", style={'color': 'red', 'fontWeight': 'bold'})

    transactions = simulation_results['transactions']
    final_balance = simulation_results['final_balance']
    balance = simulation_results['final_cash']
    positions = simulation_results['positions']
    initial_balance = simulation_results['initial_balance']
    total_profit = final_balance - initial_balance
    roi = (final_balance - initial_balance) / initial_balance * 100

    # Performance metrics
    sharpe_ratio = simulation_results['sharpe_ratio']
    max_drawdown = simulation_results['max_drawdown'] * 100  # Convert to percentage

    max_profit = simulation_results['max_profit']
    stock_symbols = simulation_results.get('stock_symbols', [])

    # Positions summary
    positions_summary = html.Ul([
        html.Li(f"{stock}: {shares} shares")
        for stock, shares in positions.items() if shares > 0
    ]) if positions else "No current holdings."

    # Earnings per stock
    earnings_per_stock = simulation_results['earnings_per_stock']
    total_shares_per_stock = simulation_results['total_shares_per_stock']

    earnings_per_share = {}
    for stock_symbol in earnings_per_stock.keys():
        total_shares = total_shares_per_stock.get(stock_symbol, 0)
        if total_shares > 0:
            eps = earnings_per_stock[stock_symbol] / total_shares
            earnings_per_share[stock_symbol] = eps
        else:
            earnings_per_share[stock_symbol] = 0

    earnings_summary = html.Ul([
        html.Li(f"{stock}: Total Earnings: {earnings_per_stock[stock]:,.2f} TL, Total Shares Traded: {total_shares_per_stock[stock]}, Earnings per Share: {eps:,.2f} TL/share")
        for stock, eps in earnings_per_share.items()
    ])

    return html.Div([
        html.H3(f"Transaction Summary"),
        html.Ul([html.Li(tx) for tx in transactions]),
        html.H4(f"Total Profit/Loss: {total_profit:,.2f} TL"),
        html.P(f"Return on Investment (ROI): {roi:.2f}%"),
        html.P(f"Final Cash Balance: {balance:,.2f} TL"),
        html.P("Positions Held:"),
        positions_summary,
        html.P(f"Total Value of the Portfolio: {final_balance:,.2f} TL"),
        html.P(f"Maximum Profit Achieved During Simulation: {max_profit:,.2f} TL"),
        html.P(f"Sharpe Ratio: {sharpe_ratio:.2f}"),
        html.P(f"Maximum Drawdown: {max_drawdown:.2f}%"),
        html.H4("Earnings per Share per Stock"),
        earnings_summary
    ], style={'borderTop': '1px solid #d9d9d9', 'paddingTop': '20px'})


if __name__ == '__main__':
    app.run_server(debug=True)
