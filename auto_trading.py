import streamlit as st
import pandas as pd
import numpy as np
import data_fetcher
import database
import trading
from datetime import datetime, timedelta
import plotly.graph_objects as go
from prediction import predict_stock_prices

def show_auto_trading_page():
    """Display the auto trading page"""
    st.title("Auto Trading")
    
    # Check if user is logged in
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in to access the auto trading features.")
        return
        
    # Get user data
    username = st.session_state.username
    user_data = database.get_user_data(username)
    
    # Display portfolio value
    portfolio_value = database.calculate_portfolio_value(user_data.get('portfolio', {}))
    st.metric("Demo Account Balance", f"${user_data.get('balance', 0):,.2f}")
    st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
    st.metric("Total Value", f"${(user_data.get('balance', 0) + portfolio_value):,.2f}")
    
    # Auto Trading Setup
    st.header("Auto Trading Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Auto trading settings
        st.subheader("Trading Rules")
        
        # Enable/disable auto trading
        auto_trading_enabled = st.toggle("Enable Auto Trading", value=user_data.get('auto_trading_enabled', False))
        
        # Risk level
        risk_options = {
            "Conservative": {
                "buy_threshold": 3.0,
                "sell_threshold": -2.0,
                "position_size": 0.1  # 10% of available funds
            },
            "Moderate": {
                "buy_threshold": 1.5,
                "sell_threshold": -1.5,
                "position_size": 0.2  # 20% of available funds
            },
            "Aggressive": {
                "buy_threshold": 0.5,
                "sell_threshold": -0.5,
                "position_size": 0.3  # 30% of available funds
            }
        }
        
        selected_risk = st.selectbox("Risk Level", list(risk_options.keys()), index=1)
        
        # Prediction model
        model_options = {
            "Linear Regression": "lr",
            "Random Forest": "rf"
        }
        selected_model = st.selectbox("Prediction Model", list(model_options.keys()))
        model_type = model_options[selected_model]
        
        # Prediction horizon
        time_horizon_options = {
            "7 days": 7,
            "14 days": 14,
            "30 days": 30
        }
        selected_horizon = st.selectbox("Prediction Horizon", list(time_horizon_options.keys()))
        days = time_horizon_options[selected_horizon]
        
        # Max positions
        max_positions = st.slider("Maximum Positions", min_value=1, max_value=20, value=user_data.get('max_positions', 5))
        
        # Maximum allocation per position
        max_allocation = st.slider("Maximum Allocation per Position (%)", min_value=1, max_value=50, value=user_data.get('max_allocation', 20))
        
        st.info("Auto trading will execute trades automatically based on model predictions and your risk settings.")
        
        # Save auto trading settings
        if st.button("Save Settings"):
            user_data['auto_trading_enabled'] = auto_trading_enabled
            user_data['risk_level'] = selected_risk
            user_data['prediction_model'] = selected_model
            user_data['prediction_horizon'] = selected_horizon
            user_data['max_positions'] = max_positions
            user_data['max_allocation'] = max_allocation
            
            # Update user data
            database.update_user_data(username, user_data)
            st.success("Auto trading settings saved!")
        
        # Manual run button
        if st.button("Run Auto Trading Once"):
            with st.spinner("Running auto trading algorithm..."):
                trades = run_auto_trading(username, model_type, days, risk_options[selected_risk], max_positions, max_allocation/100)
                if trades:
                    st.success(f"Auto trading completed! {len(trades)} trades executed.")
                else:
                    st.info("No trades were executed based on current market conditions and your settings.")
    
    with col2:
        # Performance metrics
        st.subheader("Auto Trading Performance")
        
        # Get trading history
        trading_history = database.get_trading_history_as_df(username)
        
        if trading_history is not None and not trading_history.empty:
            # Filter auto trades - make sure column exists
            if 'auto_trade' in trading_history.columns:
                auto_trades = trading_history[trading_history['auto_trade'] == True]
            else:
                auto_trades = pd.DataFrame(columns=trading_history.columns)  # Empty DataFrame with same structure
            
            if not auto_trades.empty:
                # Calculate auto trading performance
                total_auto_trades = len(auto_trades)
                buy_trades = len(auto_trades[auto_trades['order_type'] == 'buy'])
                sell_trades = len(auto_trades[auto_trades['order_type'] == 'sell'])
                
                # Calculate profit/loss for sold positions
                profit_loss = 0
                for index, row in auto_trades[auto_trades['order_type'] == 'sell'].iterrows():
                    symbol = row['symbol']
                    sell_price = row['price']
                    sell_quantity = row['quantity']
                    
                    # Find corresponding buy trades
                    buy_trades_for_symbol = auto_trades[(auto_trades['symbol'] == symbol) & 
                                                        (auto_trades['order_type'] == 'buy')]
                    
                    if not buy_trades_for_symbol.empty:
                        avg_buy_price = buy_trades_for_symbol['price'].mean()
                        profit_loss += (sell_price - avg_buy_price) * sell_quantity
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Auto Trades", total_auto_trades)
                with col2:
                    st.metric("Buy Trades", buy_trades)
                with col3:
                    st.metric("Sell Trades", sell_trades)
                
                # Profit/Loss
                st.metric("Auto Trading Profit/Loss", f"${profit_loss:.2f}", 
                           delta=f"{(profit_loss/portfolio_value*100):.2f}%" if portfolio_value > 0 else None)
                
                # Display last 5 auto trades
                st.write("**Recent Auto Trades**")
                
                display_cols = ['timestamp', 'symbol', 'order_type', 'quantity', 'price', 'total_amount']
                recent_trades = auto_trades.sort_values('timestamp', ascending=False).head(5)
                
                # Format for display
                display_df = recent_trades[display_cols].copy()
                display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
                display_df['total_amount'] = display_df['total_amount'].apply(lambda x: f"${x:.2f}")
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Rename columns
                column_names = {
                    'timestamp': 'Date',
                    'symbol': 'Symbol',
                    'order_type': 'Type',
                    'quantity': 'Quantity',
                    'price': 'Price',
                    'total_amount': 'Total'
                }
                
                display_df = display_df.rename(columns=column_names)
                
                # Apply styling to buy/sell
                def highlight_order_type(val):
                    if val == 'buy':
                        return 'background-color: green; color: white'
                    elif val == 'sell':
                        return 'background-color: red; color: white'
                    return ''
                
                st.dataframe(display_df.style.applymap(highlight_order_type, subset=['Type']), use_container_width=True)
            else:
                st.info("No auto trades have been executed yet.")
        else:
            st.info("No trading history found.")
    
    # Stock Suggestion Section
    st.header("Best Stock Suggestions")
    
    # Run stock screening to find potential opportunities
    with st.spinner("Analyzing market for best opportunities..."):
        suggestions = find_best_stocks()
        
        if suggestions:
            # Display opportunity table
            suggestion_df = pd.DataFrame(suggestions)
            
            # Format columns
            suggestion_df['current_price'] = suggestion_df['current_price'].apply(lambda x: f"${x:.2f}")
            suggestion_df['predicted_price'] = suggestion_df['predicted_price'].apply(lambda x: f"${x:.2f}")
            suggestion_df['expected_return'] = suggestion_df['expected_return'].apply(lambda x: f"{x:.2f}%")
            
            # Rename columns
            column_names = {
                'symbol': 'Symbol',
                'company_name': 'Company',
                'current_price': 'Current Price',
                'predicted_price': 'Predicted Price',
                'expected_return': 'Expected Return',
                'recommendation': 'Recommendation',
                'confidence': 'Confidence'
            }
            
            display_df = suggestion_df.rename(columns=column_names)
            
            # Apply styling to recommendations
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
                display_df.style.map(color_recommendation, subset=['Recommendation']),
                use_container_width=True
            )
            
            # Allow user to select a stock to trade
            selected_symbol = st.selectbox("Select a stock to trade:", suggestion_df['symbol'].tolist())
            
            if selected_symbol:
                # Get relevant data
                selected_stock = suggestion_df[suggestion_df['symbol'] == selected_symbol].iloc[0]
                company_name = selected_stock['company_name']
                
                # Trading form
                col1, col2 = st.columns(2)
                with col1:
                    order_type = st.radio("Order Type:", ["Buy", "Sell"], horizontal=True)
                
                with col2:
                    # Get current price
                    current_price = data_fetcher.get_current_price(selected_symbol)
                    
                    # Display current price
                    st.metric(f"{selected_symbol} Current Price", f"${current_price:.2f}")
                
                # Calculate maximum quantity based on user's balance
                max_buy_quantity = int(user_data.get('balance', 0) / current_price)
                
                # Get current holdings for the selected stock
                portfolio = user_data.get('portfolio', {})
                current_holdings = portfolio.get(selected_symbol, {}).get('quantity', 0)
                
                # Show current holdings
                st.metric(f"Current Holdings", f"{current_holdings} shares", 
                        delta=f"${current_holdings * current_price:.2f}")
                
                # Quantity slider
                if order_type == "Buy":
                    quantity = st.slider("Quantity:", min_value=1, max_value=max(1, max_buy_quantity), value=1)
                else:  # Sell
                    quantity = st.slider("Quantity:", min_value=1, max_value=max(1, current_holdings), value=min(1, current_holdings))
                
                # Calculate total
                total = quantity * current_price
                
                st.metric("Total Transaction Value", f"${total:.2f}")
                
                # Execute button
                if st.button("Execute Trade"):
                    # Check if user has enough balance for buy or enough shares for sell
                    if order_type == "Buy" and total > user_data.get('balance', 0):
                        st.error("Insufficient funds to complete this transaction.")
                    elif order_type == "Sell" and quantity > current_holdings:
                        st.error("Insufficient shares to complete this transaction.")
                    else:
                        # Execute the trade
                        result = trading.execute_order(
                            username=username,
                            symbol=selected_symbol,
                            company_name=company_name,
                            quantity=quantity,
                            price=current_price,
                            order_type=order_type.lower()
                        )
                        
                        if result:
                            st.success(f"Successfully {order_type.lower()}ed {quantity} shares of {selected_symbol} at ${current_price:.2f}")
                        else:
                            st.error("Failed to execute trade. Please try again.")


def run_auto_trading(username, model_type, days, risk_settings, max_positions, max_allocation_pct):
    """Run the auto trading algorithm"""
    # Get user data
    user_data = database.get_user_data(username)
    balance = user_data.get('balance', 0)
    portfolio = user_data.get('portfolio', {})
    
    # Get available funds to trade
    available_funds = balance
    
    # Calculate current number of positions
    current_positions = len(portfolio)
    
    # List to store executed trades
    executed_trades = []
    
    # Skip if no funds available
    if available_funds <= 100:  # Minimum $100 to trade
        return executed_trades
    
    # Get stock suggestions
    suggestions = find_best_stocks(limit=20, model_type=model_type, days=days)
    
    # Process buy signals first
    buy_candidates = [s for s in suggestions if s['recommendation'] in ['Strong Buy', 'Buy']]
    
    # Skip if no buy candidates
    if not buy_candidates:
        return executed_trades
    
    # Sort by expected return (highest first)
    buy_candidates.sort(key=lambda x: x['expected_return'], reverse=True)
    
    # Calculate max amount per position
    max_position_value = available_funds * max_allocation_pct
    
    # Execute buy trades if we have room for more positions
    if current_positions < max_positions:
        for stock in buy_candidates:
            symbol = stock['symbol']
            company_name = stock['company_name']
            current_price = data_fetcher.get_current_price(symbol)
            expected_return = stock['expected_return']
            
            # Skip if expected return is below buy threshold
            if expected_return < risk_settings['buy_threshold']:
                continue
            
            # Skip if we already own this stock
            if symbol in portfolio:
                continue
            
            # Calculate position size based on risk level
            position_value = min(max_position_value, available_funds * risk_settings['position_size'])
            
            # Calculate quantity
            quantity = int(position_value / current_price)
            
            # Skip if we can't buy at least 1 share
            if quantity < 1:
                continue
            
            # Execute the buy trade
            total_amount = quantity * current_price
            
            # Record the trade in the database
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            database.record_trade(
                username=username,
                symbol=symbol,
                company_name=company_name,
                quantity=quantity,
                price=current_price,
                total_amount=total_amount,
                order_type='buy',
                timestamp=timestamp,
                auto_trade=True
            )
            
            # Add to executed trades list
            executed_trades.append({
                'symbol': symbol,
                'order_type': 'buy',
                'quantity': quantity,
                'price': current_price,
                'total_amount': total_amount
            })
            
            # Update available funds
            available_funds -= total_amount
            
            # Update current positions count
            current_positions += 1
            
            # Stop if we've reached max positions or have low funds
            if current_positions >= max_positions or available_funds < 100:
                break
    
    # Process sell signals for existing positions
    # Refresh portfolio data after buys
    user_data = database.get_user_data(username)
    portfolio = user_data.get('portfolio', {})
    
    # Get sell candidates
    sell_candidates = []
    for symbol, stock_data in portfolio.items():
        # Get prediction for this stock
        stock_history = data_fetcher.get_stock_data(symbol, period="1y")
        if stock_history is None or stock_history.empty:
            continue
        
        # Generate predictions
        predictions, dates, accuracy = predict_stock_prices(stock_history, model_type, days)
        
        if predictions is None:
            continue
        
        current_price = stock_data.get('current_price', 0)
        last_prediction = predictions[-1]
        quantity = stock_data.get('quantity', 0)
        
        # Calculate expected return
        expected_return = ((last_prediction / current_price) - 1) * 100
        
        # Determine recommendation
        if expected_return < risk_settings['sell_threshold']:
            recommendation = "Sell"
            if expected_return < risk_settings['sell_threshold'] * 2:
                recommendation = "Strong Sell"
                
            sell_candidates.append({
                'symbol': symbol,
                'quantity': quantity,
                'current_price': current_price,
                'expected_return': expected_return,
                'recommendation': recommendation
            })
    
    # Execute sell trades
    for stock in sell_candidates:
        symbol = stock['symbol']
        quantity = stock['quantity']
        current_price = stock['current_price']
        company_name = portfolio[symbol].get('company_name', symbol)
        
        # Execute the sell trade
        total_amount = quantity * current_price
        
        # Record the trade in the database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        database.record_trade(
            username=username,
            symbol=symbol,
            company_name=company_name,
            quantity=quantity,
            price=current_price,
            total_amount=total_amount,
            order_type='sell',
            timestamp=timestamp,
            auto_trade=True
        )
        
        # Add to executed trades list
        executed_trades.append({
            'symbol': symbol,
            'order_type': 'sell',
            'quantity': quantity,
            'price': current_price,
            'total_amount': total_amount
        })
    
    return executed_trades


def find_best_stocks(limit=10, model_type="rf", days=30):
    """Find the best stock opportunities based on predictions"""
    # List of stocks to analyze (can be expanded)
    stock_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", 
                      "NFLX", "DIS", "JPM", "BAC", "WMT", "PG", "JNJ", "KO", "PEP", 
                      "MCD", "INTC", "CSCO", "IBM", "ORCL", "CRM", "ADBE", "PYPL"]
    
    suggestions = []
    
    for symbol in stock_universe:
        try:
            # Get stock data
            stock_data = data_fetcher.get_stock_data(symbol, period="1y")
            
            if stock_data is None or stock_data.empty:
                continue
            
            # Get company info
            company_info = data_fetcher.get_company_info(symbol)
            company_name = company_info.get('longName', symbol)
            
            # Generate predictions
            predictions, dates, accuracy = predict_stock_prices(stock_data, model_type, days)
            
            if predictions is None:
                continue
            
            # Get current price
            current_price = stock_data['Close'].iloc[-1]
            last_prediction = predictions[-1]
            
            # Calculate expected return
            expected_return = ((last_prediction / current_price) - 1) * 100
            
            # Determine recommendation
            if last_prediction > current_price * 1.05:  # 5% higher
                recommendation = "Strong Buy"
                confidence = "High" if accuracy > 80 else "Medium"
            elif last_prediction > current_price * 1.01:  # 1-5% higher
                recommendation = "Buy"
                confidence = "Medium" if accuracy > 70 else "Low"
            elif last_prediction < current_price * 0.95:  # 5% lower
                recommendation = "Strong Sell"
                confidence = "High" if accuracy > 80 else "Medium"
            elif last_prediction < current_price * 0.99:  # 1-5% lower
                recommendation = "Sell"
                confidence = "Medium" if accuracy > 70 else "Low"
            else:  # within 1%
                recommendation = "Hold"
                confidence = "Medium"
            
            # Add to suggestions list
            suggestions.append({
                'symbol': symbol,
                'company_name': company_name,
                'current_price': current_price,
                'predicted_price': last_prediction,
                'expected_return': expected_return,
                'recommendation': recommendation,
                'confidence': confidence,
                'accuracy': accuracy
            })
            
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {e}")
            continue
    
    # Sort by expected return for buy recommendations
    buy_suggestions = [s for s in suggestions if s['recommendation'] in ['Strong Buy', 'Buy']]
    other_suggestions = [s for s in suggestions if s['recommendation'] not in ['Strong Buy', 'Buy']]
    
    buy_suggestions.sort(key=lambda x: x['expected_return'], reverse=True)
    other_suggestions.sort(key=lambda x: x['expected_return'])
    
    # Combine the lists with buy recommendations first
    sorted_suggestions = buy_suggestions + other_suggestions
    
    # Limit the number of suggestions
    return sorted_suggestions[:limit]