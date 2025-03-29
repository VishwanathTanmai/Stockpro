import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import data_fetcher
import database

def show_prediction_page():
    """Display the stock prediction page"""
    if not st.session_state.logged_in:
        st.warning("Please login to access prediction features.")
        return
    
    st.header("Stock Price Prediction")
    
    # Input for stock symbol
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input("Enter stock symbol (e.g. AAPL, MSFT, GOOGL)", key="prediction_symbol_input")
    
    with col2:
        if st.button("Analyze", key="analyze_stock_button"):
            if symbol:
                st.session_state.stock_to_predict = symbol.upper()
            else:
                st.warning("Please enter a stock symbol.")
    
    # If a stock symbol is provided, show predictions
    if 'stock_to_predict' in st.session_state and st.session_state.stock_to_predict:
        symbol = st.session_state.stock_to_predict
        stock_data = data_fetcher.get_stock_data(symbol, period="1y")
        
        if stock_data is not None:
            # Display company name
            company_info = data_fetcher.get_company_info(symbol)
            company_name = company_info.get('longName', symbol)
            
            st.subheader(f"Analysis for {company_name} ({symbol})")
            
            # Get extended historical data for the stock
            historical_data = data_fetcher.get_historical_stock_data(symbol, period="max")
            
            # Historical data section
            with st.expander("Historical Data Analysis", expanded=True):
                # Time period selector for historical data
                period_options = {
                    "1 Week": 7, 
                    "1 Month": 30, 
                    "3 Months": 90, 
                    "6 Months": 180, 
                    "1 Year": 365,
                    "3 Years": 365*3,
                    "5 Years": 365*5,
                    "10 Years": 365*10,
                    "Max": 0  # 0 means all available data
                }
                
                selected_period = st.selectbox("Select time period:", list(period_options.keys()), index=4)
                days = period_options[selected_period]
                
                # Filter data based on selected period
                if days > 0 and historical_data is not None:
                    # Convert start_date to timezone-naive pandas Timestamp to avoid timezone issues
                    start_date = pd.Timestamp(datetime.now() - timedelta(days=days)).tz_localize(None)
                    
                    # Convert historical_data index to timezone-naive for comparison if it has timezone info
                    if historical_data.index.tz is not None:
                        compare_index = historical_data.index.tz_localize(None)
                        filtered_data = historical_data[compare_index >= start_date]
                    else:
                        filtered_data = historical_data[historical_data.index >= start_date]
                else:
                    filtered_data = historical_data
                
                if filtered_data is not None and not filtered_data.empty:
                    # Show descriptive statistics
                    st.write(f"**Price Statistics ({selected_period})**")
                    
                    stats_df = pd.DataFrame({
                        'Statistic': ['Current Price', 'Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                        'Value': [
                            filtered_data['Close'].iloc[-1],
                            filtered_data['Close'].mean(),
                            filtered_data['Close'].median(),
                            filtered_data['Close'].min(),
                            filtered_data['Close'].max(),
                            filtered_data['Close'].std()
                        ]
                    })
                    
                    stats_df['Value'] = stats_df['Value'].apply(lambda x: f"${x:.2f}")
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Calculate returns for different periods
                    returns_df = pd.DataFrame({
                        'Period': ['1 Day', '1 Week', '1 Month', '3 Months', '6 Months', '1 Year', 'YTD'],
                        'Return (%)': [
                            ((filtered_data['Close'].iloc[-1] / filtered_data['Close'].iloc[-2]) - 1) * 100 if len(filtered_data) > 1 else 0,
                            ((filtered_data['Close'].iloc[-1] / filtered_data['Close'].iloc[-min(5, len(filtered_data))]) - 1) * 100 if len(filtered_data) >= 5 else 0,
                            ((filtered_data['Close'].iloc[-1] / filtered_data['Close'].iloc[-min(21, len(filtered_data))]) - 1) * 100 if len(filtered_data) >= 21 else 0,
                            ((filtered_data['Close'].iloc[-1] / filtered_data['Close'].iloc[-min(63, len(filtered_data))]) - 1) * 100 if len(filtered_data) >= 63 else 0,
                            ((filtered_data['Close'].iloc[-1] / filtered_data['Close'].iloc[-min(126, len(filtered_data))]) - 1) * 100 if len(filtered_data) >= 126 else 0,
                            ((filtered_data['Close'].iloc[-1] / filtered_data['Close'].iloc[-min(252, len(filtered_data))]) - 1) * 100 if len(filtered_data) >= 252 else 0,
                            ((filtered_data['Close'].iloc[-1] / filtered_data['Close'][filtered_data.index.to_series().dt.year == datetime.now().year].iloc[0]) - 1) * 100 if not filtered_data[filtered_data.index.to_series().dt.year == datetime.now().year].empty else 0
                        ]
                    })
                    
                    # Color code returns
                    def highlight_returns(val):
                        if isinstance(val, float):
                            if val > 0:
                                return f'color: green'
                            elif val < 0:
                                return f'color: red'
                        return ''
                    
                    # Format the returns
                    returns_df['Return (%)'] = returns_df['Return (%)'].apply(lambda x: f"{x:.2f}%")
                    
                    st.write("**Returns by Period**")
                    st.dataframe(returns_df.style.applymap(highlight_returns, subset=['Return (%)']), use_container_width=True)
                    
                    # Historical price chart
                    st.write("**Historical Price Chart**")
                    
                    # Choose chart type
                    chart_type = st.radio("Chart Type:", ["Line", "Candlestick"], horizontal=True)
                    
                    if chart_type == "Line":
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=filtered_data.index,
                            y=filtered_data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='royalblue', width=2)
                        ))
                        
                        # Add moving averages
                        for window in [20, 50, 200]:
                            if len(filtered_data) > window:
                                filtered_data[f'MA{window}'] = filtered_data['Close'].rolling(window=window).mean()
                                fig.add_trace(go.Scatter(
                                    x=filtered_data.index,
                                    y=filtered_data[f'MA{window}'],
                                    mode='lines',
                                    name=f'{window}-day MA',
                                    line=dict(width=1.5)
                                ))
                    else:  # Candlestick
                        fig = go.Figure(data=[go.Candlestick(
                            x=filtered_data.index,
                            open=filtered_data['Open'],
                            high=filtered_data['High'],
                            low=filtered_data['Low'],
                            close=filtered_data['Close'],
                            name='Price'
                        )])
                    
                    fig.update_layout(
                        height=500,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis_title='Price (USD)',
                        xaxis_title='Date',
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show volume
                    st.write("**Trading Volume**")
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(
                        x=filtered_data.index,
                        y=filtered_data['Volume'],
                        name='Volume',
                        marker_color='rgba(0, 0, 255, 0.3)'
                    ))
                    
                    fig_volume.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis_title='Volume',
                        xaxis_title='Date',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
                    
                    # Calculate returns
                    filtered_data['Daily Return'] = filtered_data['Close'].pct_change() * 100
                    
                    # Daily returns chart
                    st.write("**Daily Returns**")
                    
                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Bar(
                        x=filtered_data.index,
                        y=filtered_data['Daily Return'],
                        name='Daily Return',
                        marker_color=filtered_data['Daily Return'].apply(lambda x: 'green' if x >= 0 else 'red')
                    ))
                    
                    fig_returns.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis_title='Return (%)',
                        xaxis_title='Date',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_returns, use_container_width=True)
                else:
                    st.error("Could not retrieve historical data for this period.")
            
            # Prediction section
            st.subheader("Price Predictions")
            
            # Select prediction horizon
            days_options = {
                "7 days": 7, 
                "14 days": 14, 
                "30 days": 30, 
                "3 months": 90, 
                "6 months": 180, 
                "1 year": 365
            }
            selected_days = st.selectbox("Select prediction horizon:", list(days_options.keys()))
            days = days_options[selected_days]
            
            # ML model selection
            model_options = {
                "Linear Regression": "lr",
                "Random Forest": "rf",
                "Neural Network": "nn"
            }
            selected_model = st.selectbox("Select prediction model:", list(model_options.keys()))
            model_type = model_options[selected_model]
            
            # Advanced options for Neural Network
            if model_type == "nn":
                st.info("Neural Networks can capture complex patterns in time series data and often provide more accurate predictions for stock prices.")
                with st.expander("Advanced Neural Network Options"):
                    hidden_layers = st.slider("Hidden Layers", min_value=1, max_value=5, value=2)
                    neurons = st.slider("Neurons per Layer", min_value=10, max_value=200, value=100, step=10)
                    epochs = st.slider("Training Epochs", min_value=50, max_value=1000, value=200, step=50)
                    learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
            else:
                hidden_layers = 2
                neurons = 100
                epochs = 200
                learning_rate = 0.01
            
            if st.button("Generate Predictions", key="generate_predictions"):
                with st.spinner(f"Generating {days}-day predictions using {selected_model}..."):
                    # Generate predictions
                    predictions, dates, accuracy = predict_stock_prices(stock_data, model_type, days, 
                                                                      hidden_layers=hidden_layers,
                                                                      neurons=neurons,
                                                                      epochs=epochs,
                                                                      learning_rate=learning_rate)
                    
                    if predictions is not None:
                        # Display the predicted prices
                        st.write(f"**Predicted Prices for Next {days} Days**")
                        
                        # Create visualization
                        fig_pred = go.Figure()
                        
                        # Historical data
                        fig_pred.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'],
                            mode='lines',
                            name='Historical Price',
                            line=dict(color='royalblue', width=2)
                        ))
                        
                        # Predictions
                        fig_pred.add_trace(go.Scatter(
                            x=dates,
                            y=predictions,
                            mode='lines+markers',
                            name='Predicted Price',
                            line=dict(color='green', width=2, dash='dash'),
                            marker=dict(size=8)
                        ))
                        
                        fig_pred.update_layout(
                            height=500,
                            margin=dict(l=0, r=0, t=0, b=0),
                            yaxis_title='Price (USD)',
                            xaxis_title='Date',
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Display model accuracy
                        st.metric("Model Accuracy", f"{accuracy:.2f}%")
                        
                        # Create a dataframe of predictions
                        pred_df = pd.DataFrame({
                            'Date': dates,
                            'Predicted Price': predictions
                        })
                        
                        pred_df['Date'] = pred_df['Date'].dt.strftime('%Y-%m-%d')
                        pred_df['Predicted Price'] = pred_df['Predicted Price'].apply(lambda x: f"${x:.2f}")
                        
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Trading recommendation
                        current_price = stock_data['Close'].iloc[-1]
                        last_prediction = predictions[-1]
                        
                        st.subheader("Trading Recommendation")
                        
                        if last_prediction > current_price * 1.05:  # 5% higher
                            recommendation = "Strong Buy"
                            color = "green"
                        elif last_prediction > current_price * 1.01:  # 1-5% higher
                            recommendation = "Buy"
                            color = "lightgreen"
                        elif last_prediction < current_price * 0.95:  # 5% lower
                            recommendation = "Strong Sell"
                            color = "red"
                        elif last_prediction < current_price * 0.99:  # 1-5% lower
                            recommendation = "Sell"
                            color = "lightcoral"
                        else:  # within 1%
                            recommendation = "Hold"
                            color = "gray"
                        
                        st.markdown(f"""
                        <div style='background-color: {color}; padding: 10px; border-radius: 5px;'>
                            <h3 style='text-align: center; color: white;'>{recommendation}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        expected_change = ((last_prediction / current_price) - 1) * 100
                        st.write(f"Expected change in {days} days: {expected_change:.2f}%")
                        
                        # Save prediction to history
                        save_prediction(
                            username=st.session_state.username,
                            symbol=symbol,
                            company_name=company_name,
                            current_price=current_price,
                            prediction=last_prediction,
                            days=days,
                            model=selected_model,
                            recommendation=recommendation
                        )
            
            # Previous predictions section
            with st.expander("Your Previous Predictions", expanded=False):
                show_prediction_history(st.session_state.username)
        else:
            st.error(f"Could not find data for stock symbol: {symbol}")

def predict_stock_prices(stock_data, model_type, days, hidden_layers=2, neurons=100, epochs=200, learning_rate=0.01):
    """Generate stock price predictions using ML models"""
    # Feature engineering
    data = stock_data.copy()
    
    # Create features
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=20).std()
    
    # Drop NaN values
    data = data.dropna()
    
    if len(data) < 60:  # Not enough data for reliable prediction
        st.warning("Not enough historical data for reliable prediction.")
        return None, None, 0
    
    # Prepare features and target
    features = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'MA50', 'Volatility']
    X = data[features].values
    y = data['Close'].values
    
    # Scale the features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    # Split into training and testing sets
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
    
    # Different model training approaches based on selected model
    if model_type == 'nn':
        # Create neural network architecture based on parameters
        hidden_layer_sizes = tuple([neurons] * hidden_layers)
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom callback for training progress
        class ProgressCallback:
            def __init__(self, total_epochs):
                self.total_epochs = total_epochs
                self.current_epoch = 0
            
            def __call__(self, estimator, locals, globals):
                self.current_epoch += 1
                progress = self.current_epoch / self.total_epochs
                progress_bar.progress(progress)
                status_text.text(f"Training Neural Network: {self.current_epoch}/{self.total_epochs} epochs completed")
                return False  # Continue training
        
        # Create progress callback
        progress_callback = ProgressCallback(epochs)
        
        # Create the neural network model
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate_init=learning_rate,
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False,
            warm_start=False
        )
        
        # Set the callback manually
        model._check_n_iter = progress_callback
        
        # Train the model
        model.fit(X_train, y_train.ravel())
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test.ravel()) ** 2)
        accuracy = (1 - mse) * 100
        
        # Make future predictions
        last_data = data.iloc[-1]
        predictions = []
        dates = []
        
        current_date = data.index[-1]
        for i in range(1, days + 1):
            # Get the last data point
            last_features = last_data[features].values.reshape(1, -1)
            last_features_scaled = scaler_X.transform(last_features)
            
            # Predict the next day
            next_price_scaled = model.predict(last_features_scaled).reshape(-1, 1)
            next_price = scaler_y.inverse_transform(next_price_scaled)[0][0]
            
            # Update dates
            next_date = current_date + timedelta(days=i)
            while next_date.weekday() > 4:  # Skip weekends
                next_date = next_date + timedelta(days=1)
            
            # Save the prediction
            predictions.append(next_price)
            dates.append(next_date)
            
            # Update the last data for the next iteration
            # Here we're making a simple assumption that the next day's Open is the same as previous day's Close
            last_data_dict = last_data.to_dict()
            last_data_dict['Open'] = next_price
            last_data_dict['High'] = next_price * 1.01  # Simple assumption
            last_data_dict['Low'] = next_price * 0.99   # Simple assumption
            last_data_dict['Close'] = next_price
            last_data_dict['MA5'] = (last_data_dict['MA5'] * 5 - data['Close'].iloc[-5] + next_price) / 5
            last_data_dict['MA20'] = (last_data_dict['MA20'] * 20 - data['Close'].iloc[-20] + next_price) / 20
            last_data_dict['MA50'] = (last_data_dict['MA50'] * 50 - data['Close'].iloc[-50] + next_price) / 50
            last_data_dict['Return'] = (next_price / last_data_dict['Close'] - 1)
            last_data_dict['Volatility'] = last_data_dict['Volatility']  # Keep the same for simplicity
            
            # Convert to Series and update
            last_data = pd.Series(last_data_dict)
            
    else:
        # Traditional ML approaches (Linear Regression or Random Forest)
        if model_type == 'lr':
            model = LinearRegression()
        else:  # Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train.ravel())
        
        # Calculate model accuracy
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_test.ravel()) ** 2)
        accuracy = (1 - mse) * 100  # Simple way to convert MSE to "accuracy"
        
        # Make future predictions
        last_data = data.iloc[-1]
        predictions = []
        dates = []
        
        current_date = data.index[-1]
        for i in range(1, days + 1):
            # Get the last data point
            last_features = last_data[features].values.reshape(1, -1)
            last_features_scaled = scaler_X.transform(last_features)
            
            # Predict the next day
            next_price_scaled = model.predict(last_features_scaled)
            next_price = scaler_y.inverse_transform(next_price_scaled.reshape(-1, 1))[0][0]
            
            # Update dates
            next_date = current_date + timedelta(days=i)
            while next_date.weekday() > 4:  # Skip weekends
                next_date = next_date + timedelta(days=1)
            
            # Save the prediction
            predictions.append(next_price)
            dates.append(next_date)
            
            # Update the last data for the next iteration
            # Here we're making a simple assumption that the next day's Open is the same as previous day's Close
            last_data_dict = last_data.to_dict()
            last_data_dict['Open'] = next_price
            last_data_dict['High'] = next_price * 1.01  # Simple assumption
            last_data_dict['Low'] = next_price * 0.99   # Simple assumption
            last_data_dict['Close'] = next_price
            last_data_dict['MA5'] = (last_data_dict['MA5'] * 5 - data['Close'].iloc[-5] + next_price) / 5
            last_data_dict['MA20'] = (last_data_dict['MA20'] * 20 - data['Close'].iloc[-20] + next_price) / 20
            last_data_dict['MA50'] = (last_data_dict['MA50'] * 50 - data['Close'].iloc[-50] + next_price) / 50
            last_data_dict['Return'] = (next_price / last_data_dict['Close'] - 1)
            last_data_dict['Volatility'] = last_data_dict['Volatility']  # Keep the same for simplicity
            
            last_data = pd.Series(last_data_dict)
    
    return np.array(predictions), pd.DatetimeIndex(dates), accuracy

def save_prediction(username, symbol, company_name, current_price, prediction, days, model, recommendation):
    """Save a prediction to the database"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    expected_change = ((prediction / current_price) - 1) * 100
    
    prediction_data = {
        'timestamp': timestamp,
        'symbol': symbol,
        'company_name': company_name,
        'current_price': current_price,
        'predicted_price': prediction,
        'days': days,
        'model': model,
        'expected_change': expected_change,
        'recommendation': recommendation
    }
    
    database.save_prediction(username, prediction_data)

def show_prediction_history(username):
    """Show the user's prediction history"""
    predictions = database.get_prediction_history(username)
    
    if predictions:
        # Convert to dataframe for display
        pred_df = pd.DataFrame(predictions)
        pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
        pred_df = pred_df.sort_values('timestamp', ascending=False)
        
        # Format for display
        display_df = pred_df.copy()
        display_df['current_price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
        display_df['predicted_price'] = display_df['predicted_price'].apply(lambda x: f"${x:.2f}")
        display_df['expected_change'] = display_df['expected_change'].apply(lambda x: f"{x:.2f}%")
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Rename columns for better display
        column_names = {
            'timestamp': 'Date',
            'symbol': 'Symbol',
            'company_name': 'Company',
            'current_price': 'Price at Prediction',
            'predicted_price': 'Predicted Price',
            'days': 'Days',
            'model': 'Model',
            'expected_change': 'Expected Change',
            'recommendation': 'Recommendation'
        }
        
        display_df = display_df.rename(columns=column_names)
        
        # Ensure we only show specific columns in a specific order
        display_columns = ['Date', 'Symbol', 'Company', 'Price at Prediction', 
                          'Predicted Price', 'Days', 'Expected Change', 'Recommendation']
        display_df = display_df[display_columns]
        
        # Apply styling based on recommendation
        def color_recommendation(val):
            if 'Strong Buy' in val:
                return 'background-color: green; color: white'
            elif 'Buy' in val:
                return 'background-color: lightgreen'
            elif 'Strong Sell' in val:
                return 'background-color: red; color: white'
            elif 'Sell' in val:
                return 'background-color: lightcoral'
            elif 'Hold' in val:
                return 'background-color: gray; color: white'
            return ''
        
        st.dataframe(
            display_df.style.applymap(color_recommendation, subset=['Recommendation']),
            use_container_width=True
        )
    else:
        st.info("No prediction history found. Make some predictions to see them here!")
