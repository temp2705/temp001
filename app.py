import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os


warnings.filterwarnings("ignore")


# Function to check if user_details.csv exists and has the correct columns
def initialize_user_details():
    if not os.path.exists('user_details.csv'):
        # If the file does not exist, create it with the correct columns
        df = pd.DataFrame(columns=['username', 'password', 'backtest_count'])
        df.to_csv('user_details.csv', index=False)
    else:
        # Ensure the file has the correct columns
        df = pd.read_csv('user_details.csv')
        if 'backtest_count' not in df.columns:
            df['backtest_count'] = 0
            df.to_csv('user_details.csv', index=False)

# Function to check login
def check_login(username, password):
    user_details = pd.read_csv('user_details.csv')
    user = user_details[(user_details['username'] == username) & (user_details['password'] == password)]
    if not user.empty:
        st.session_state.backtest_count = user['backtest_count'].values[0]
        return True
    return False

# Function to register a new user
def register_user(username, password):
    user_details = pd.read_csv('user_details.csv')
    if username in user_details['username'].values:
        return False
    new_user = pd.DataFrame({'username': [username], 'password': [password], 'backtest_count': [0]})
    new_user.to_csv('user_details.csv', mode='a', header=False, index=False)
    return True

# Function to update the backtest count for a user
def update_backtest_count(username, count):
    user_details = pd.read_csv('user_details.csv')
    user_details.loc[user_details['username'] == username, 'backtest_count'] = count
    user_details.to_csv('user_details.csv', index=False)

# Initialize user_details.csv if it doesn't exist
initialize_user_details()

# Set page title and description
st.set_page_config(page_title='Stock Analysis Tool', page_icon=':chart_with_upwards_trend:')
st.title('Stock Analysis Tool')
st.write('This app analyzes historical stock data based on user-defined criteria.')

# Define custom CSS styling for the boxes
box_style = """
    background-color: #2f4f4f;
    color: white;
    text-align: center;
    font-size: 18px;
    padding: 20px;
    margin: 10px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
"""

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.backtest_count = 0

# Login/Register form
if not st.session_state.logged_in:
    login_expander = st.expander("Login")
    with login_expander:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")
        #st.write("Don't have an account? Click below to register.")
                
                
else:
    st.sidebar.header(f"Welcome, {st.session_state.username}!")
    st.sidebar.write(f"You have {5 - st.session_state.backtest_count} backtests remaining.")


    # Function to limit backtests
    if st.session_state.backtest_count < 5:
        # Sidebar for user inputs
        st.sidebar.header('Input Parameters')

        # Input fields in the sidebar
        ticker_symbol = st.sidebar.text_input("Enter the ticker symbol (e.g., 'BEML.NS')")
        trade_type = st.sidebar.selectbox("Select trade type", ['delivery', 'intraday'])
        target_level = st.sidebar.slider("Select target level (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)

        # Available timeframes and their corresponding maximum days of historical data
        timeframes = {
            '1m': 7,
            '2m': 60,
            '5m': 60,
            '15m': 60,
            '30m': 60,
            '1h': 730,
            '1d': 'unlimited',
            '1wk': 'unlimited',
            '1mo': 'unlimited'
        }

        # Dropdown for selecting the timeframe
        timeframe = st.sidebar.selectbox("Select timeframe", list(timeframes.keys()))

        bullish_definition = st.sidebar.selectbox("Select bullish consecutive candles",
                                                ['Close > Open', 'Next candle low didn\'t break previous candle low', 'Both'])
        use_rsi = st.sidebar.radio("Use RSI filter", ['yes', 'no'])

        if use_rsi == 'yes':
            rsi_level = st.sidebar.slider("Select RSI level to filter", min_value=50, max_value=100, value=70, step=1)

        # Additional inputs based on timeframe
        if timeframe.endswith('m') or timeframe == '1h':
            period_days = st.sidebar.slider(f"Select number of days for backtesting (Max: {timeframes[timeframe]} days)", 
                                            min_value=1, max_value=timeframes[timeframe], value=min(30, timeframes[timeframe]))
            period = f'{period_days}d'
        else:
            start_date = st.sidebar.date_input("Select start date", value=pd.to_datetime('2021-01-01'))
            end_date = pd.to_datetime('today').date()  # End date defaults to today

        num_bullish_candles = st.sidebar.slider("Select number of consecutive bullish candles required", min_value=1, max_value=10, value=6)

        # Function to calculate RSI
        def calculate_rsi(data, window=14):
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        import plotly.graph_objects as go

        # Function to plot the gauge chart
        def plot_gauge_chart(win_rate):
            # Determine color based on win_rate
            if win_rate >= 75:
                color = 'green'
            else:
                color = 'red'
            
            # Define the figure for the gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = win_rate,  # Set the value to win_rate
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Win Rate"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                }
            ))

            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))

            return fig


        # Enhanced Function to plot drawdown
        def plot_drawdown(df):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df['drawdown'], label='Drawdown', color='red', linewidth=1.5)
            ax.fill_between(df.index, df['drawdown'], color='red', alpha=0.3)
            ax.set_title('Drawdown Analysis', fontsize=16)
            ax.set_xlabel('Date', fontsize=14)
            ax.set_ylabel('Drawdown (%)', fontsize=14)
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            st.pyplot(fig)

        # Enhanced Function to calculate equity and drawdown
        def calculate_equity_drawdown(df):
            equity = df['P&L %'].cumsum().to_frame(name='Profit/Loss %')
            equity['Drawdown %'] = equity['Profit/Loss %'] - equity['Profit/Loss %'].cummax()
            return equity

        # Enhanced Function to plot equity and drawdown
        def plot_equity_drawdown(equity):
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Equity Analysis", "Drawdown"), vertical_spacing=0.1)
            
            fig.add_trace(go.Scatter(
                x=equity.index, 
                y=equity['Profit/Loss %'], 
                mode='lines', 
                name='Equity', 
                fill='tozeroy', 
                line=dict(color='green', width=1.5)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=equity.index, 
                y=equity['Drawdown %'], 
                mode='lines', 
                name='Drawdown', 
                fill='tozeroy', 
                line=dict(color='firebrick', width=1.5)
            ), row=2, col=1)
            
            fig.update_layout(
                title='Equity Analysis and Drawdown',
                xaxis=dict(title='Date', titlefont_size=14, tickfont_size=12),
                yaxis=dict(title='Profit/Loss %', titlefont_size=14, tickfont_size=12),
                yaxis2=dict(title='Drawdown %', titlefont_size=14, tickfont_size=12),
                template='plotly_dark',
                height=800
            )
            
            return fig

        # Function to download data and perform analysis
        def analyze_stock_data(ticker_symbol, trade_type, target_level, timeframe, bullish_definition, use_rsi, rsi_level, start_date=None, end_date=None, num_bullish_candles=6):
            # Download historical data
            if timeframe.endswith('m') or timeframe == '1h':
                data = yf.download(ticker_symbol, period=period, interval=timeframe)
            else:
                data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=timeframe)

            # Filter out only the necessary columns
            data = data[['Open', 'Close', 'Low', 'High']]

            # Ensure the index of data is datetime-like
            data.index = pd.to_datetime(data.index)

            if trade_type == 'intraday':
                # Filter for intraday data only (9:15 to 15:30)
                data = data.between_time('09:15', '15:30')

            # Calculate RSI function
            def calculate_rsi(data, window=14):
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            # Add RSI column to the data DataFrame
            data['RSI'] = calculate_rsi(data)

            # Function to process the end of a bullish streak
            def process_bullish_streak():
                nonlocal current_bullish_streak, current_bullish_high, last_bullish_low

                if current_bullish_streak >= num_bullish_candles:  # Only add streaks of specified number of candles or more
                    # Add the details of the last bullish candle to the list
                    last_bullish_index = i - 1
                    last_bullish_date = data.index[last_bullish_index]
                    last_bullish_low = data['Low'][last_bullish_index]
                    last_bullish_rsi = data['RSI'][last_bullish_index]
                    target_price = last_bullish_low * (1 - target_level / 100)  # Target based on user input

                    # Apply RSI filter if selected
                    if use_rsi == 'yes' and last_bullish_rsi < rsi_level:
                        return

                    # Find the number of candles after the bullish streak where low breaks last bullish low
                    entry_time = None
                    exit_time = None
                    exit_price = None
                    candles_to_break_low = 0
                    break_low_value = None
                    break_high_value = None
                    candles_to_achieve_target_or_stoploss = 0
                    result = None

                    for j in range(i, len(data)):
                        if trade_type == 'intraday' and data.index[j].date() != last_bullish_date.date():
                            break  # Stop if the date changes (end of intraday session)
                        if data['Low'][j] < last_bullish_low:
                            candles_to_break_low = j - last_bullish_index
                            break_low_value = data['Low'][j]
                            break_high_value = data['High'][j]
                            entry_time = data.index[j]

                            for k in range(j + 1, len(data)):
                                if trade_type == 'intraday' and data.index[k].date() != last_bullish_date.date():
                                    break  # Stop if the date changes (end of intraday session)
                                candles_to_achieve_target_or_stoploss += 1
                                if data['Low'][k] <= target_price and data['High'][k] >= current_bullish_high:
                                    exit_time = data.index[k]
                                    exit_price = min(target_price, current_bullish_high)
                                    result = 'Both in single candle'
                                    break
                                elif data['Low'][k] <= target_price:
                                    exit_time = data.index[k]
                                    exit_price = target_price
                                    result = 'Target Achieved'
                                    break
                                elif data['High'][k] >= current_bullish_high:
                                    exit_time = data.index[k]
                                    exit_price = current_bullish_high
                                    result = 'Stop Loss Hit'
                                    break
                            break

                    # Calculate returns and P&L %
                    if result == 'Target Achieved':
                        returns = last_bullish_low - exit_price
                        p_and_l = (returns / last_bullish_low) * 100
                    elif result == 'Stop Loss Hit':
                        returns = last_bullish_low - exit_price
                        p_and_l = (returns / last_bullish_low) * 100
                    elif result == 'Both in single candle':
                        returns = 0
                        p_and_l = 0
                    else:
                        returns = None
                        p_and_l = None

                    # Add trade details to consecutive_bullish_candles
                    consecutive_bullish_candles.append({
                        'Date': last_bullish_date.date(),
                        'Entry Time': entry_time.time() if entry_time else None,
                        'Exit Time': exit_time.time() if exit_time else None,
                        'Last Bullish Candle Time': data.index[i - 1].time(),  # Time of the last consecutive bullish candle formed
                        'Entry Price': last_bullish_low,
                        'Stop Loss': current_bullish_high,
                        'Number of Candles': current_bullish_streak,
                        'RSI': last_bullish_rsi,
                        'Candles to Break Low': candles_to_break_low,
                        'Break Low Value': break_low_value,
                        'Break High Value': break_high_value,
                        'Candles to Achieve Target or Stoploss': candles_to_achieve_target_or_stoploss,
                        'Target': target_price,
                        'Exit Price': exit_price,
                        'Returns': returns,
                        'P&L %': p_and_l,
                        'Result': result
                    })

                # Reset streak variables after adding trade details
                current_bullish_streak = 0
                current_bullish_high = None
                last_bullish_low = None  # Reset last_bullish_low

            # Determine consecutive bullish candles based on user's choice
            consecutive_bullish_candles = []
            current_bullish_streak = 0
            current_bullish_high = None
            last_bullish_low = None

            for i in range(1, len(data)):
                if bullish_definition == 'Close > Open':  # Close > Open
                    if data['Close'][i] > data['Open'][i]:
                        current_bullish_streak += 1
                        if current_bullish_high is None or data['High'][i] > current_bullish_high:
                            current_bullish_high = data['High'][i]
                        if last_bullish_low is None or data['Low'][i] < last_bullish_low:
                            last_bullish_low = data['Low'][i]  # Update last_bullish_low
                    else:
                        process_bullish_streak()
                elif bullish_definition == 'Next candle low didn\'t break previous candle low':  # Next candle low didn't break previous candle low
                    if data['Close'][i] > data['Open'][i] and data['Low'][i] >= last_bullish_low:
                        current_bullish_streak += 1
                        if current_bullish_high is None or data['High'][i] > current_bullish_high:
                            current_bullish_high = data['High'][i]
                        if last_bullish_low is None or data['Low'][i] < last_bullish_low:
                            last_bullish_low = data['Low'][i]  # Update last_bullish_low
                    else:
                        process_bullish_streak()
                elif bullish_definition == 'Both':  # Both conditions
                    if data['Close'][i] > data['Open'][i] and (last_bullish_low is None or data['Low'][i] >= last_bullish_low):
                        current_bullish_streak += 1
                        if current_bullish_high is None or data['High'][i] > current_bullish_high:
                            current_bullish_high = data['High'][i]
                        if last_bullish_low is None or data['Low'][i] < last_bullish_low:
                            last_bullish_low = data['Low'][i]  # Update last_bullish_low
                    else:
                        process_bullish_streak()

            # Final streak check at the end of the loop
            process_bullish_streak()

            # Convert the list of consecutive bullish candles into a DataFrame
            df_consecutive_bullish = pd.DataFrame(consecutive_bullish_candles)

            # Check if df_consecutive_bullish is empty before filtering
            if not df_consecutive_bullish.empty:
                # Filter out entries where no candle broke the low
                df_consecutive_bullish = df_consecutive_bullish[df_consecutive_bullish['Candles to Break Low'] > 0]

                # Calculate important KPIs
                total_trades = len(df_consecutive_bullish)
                winning_trades = len(df_consecutive_bullish[df_consecutive_bullish['P&L %'] > 0])
                losing_trades = len(df_consecutive_bullish[df_consecutive_bullish['P&L %'] < 0])
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                average_return = df_consecutive_bullish['P&L %'].mean() if len(df_consecutive_bullish) > 0 else 0
                max_return = df_consecutive_bullish['P&L %'].max() if len(df_consecutive_bullish) > 0 else 0
                min_return = df_consecutive_bullish['P&L %'].min() if len(df_consecutive_bullish) > 0 else 0
                total_return = df_consecutive_bullish['P&L %'].sum() if len(df_consecutive_bullish) > 0 else 0

                st.subheader("Key Metrics")

                chart = plot_gauge_chart(win_rate)
                st.plotly_chart(chart)
                # Display key metrics with improved styling
                st.markdown(
                    '<div style="display: flex; flex-wrap: wrap; justify-content: space-around;">'
                    f'<div style="{box_style}; flex: 25%;">Number of Trades:<br><b>{total_trades:.2f}</b></div>'
                    f'<div style="{box_style}; flex: 25%;">Total Successful Trades:<br><b>{winning_trades:.2f}</b></div>'
                    f'<div style="{box_style}; flex: 25%;">Total Failure Trades:<br><b>{losing_trades:.2f}</b></div>'
                    f'<div style="{box_style}; flex: 25%;">Profitability (Win Rate):<br><b>{win_rate:.2f}%</b></div>'
                    f'<div style="{box_style}; flex: 25%;">Overall P&L Returns:<br><b>{total_return:.2f}%</b></div>'
                    '</div>',
                    unsafe_allow_html=True
                )

                # Filter out entries where no candle broke the low
                df_consecutive_bullish = df_consecutive_bullish[df_consecutive_bullish['Candles to Break Low'] > 0]
                equity = calculate_equity_drawdown(df_consecutive_bullish)
                drawdown = equity['Drawdown %']

                #st.subheader('Consecutive Bullish Candles Analysis')
                #st.write(df_consecutive_bullish)
                st.plotly_chart(plot_equity_drawdown(equity))

                # Increment the backtest count
                st.session_state.backtest_count += 1
                st.sidebar.write(f"You have {5 - st.session_state.backtest_count} backtests left.")
                

                # Display the filtered DataFrame
                if len(df_consecutive_bullish) > 0:
                    st.write("\nFiltered DataFrame:")
                    st.write(df_consecutive_bullish)
                else:
                    st.write("No trades found based on the criteria.")
            else:
                st.write("No trades found based on the criteria.")



        if st.sidebar.button('Backtest'):
            if timeframe.endswith('m'):
                analyze_stock_data(ticker_symbol, trade_type, target_level, timeframe, bullish_definition, use_rsi, rsi_level, num_bullish_candles=num_bullish_candles)
            else:
                analyze_stock_data(ticker_symbol, trade_type, target_level, timeframe, bullish_definition, use_rsi, rsi_level, start_date=start_date, end_date=end_date, num_bullish_candles=num_bullish_candles)

    else:
        st.sidebar.error("You have reached the maximum number of backtests allowed.")
