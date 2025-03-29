import streamlit as st
import pandas as pd
import json
import os
import hashlib
from datetime import datetime

# In a real application, you would use a proper database like SQLite, PostgreSQL, etc.
# For this demo, we'll use session_state for in-memory storage and simulate persistence

def initialize_db():
    """Initialize the database structures if they don't exist"""
    # Users database
    if 'users_db' not in st.session_state:
        st.session_state.users_db = {}
    
    # Load from file if exists (to simulate persistence)
    try:
        with open('users_db.json', 'r') as f:
            st.session_state.users_db = json.load(f)
    except FileNotFoundError:
        # Create example user
        create_user('demo', 'demo@example.com', 'demo123')

def save_db():
    """Save the database to file (to simulate persistence)"""
    try:
        with open('users_db.json', 'w') as f:
            json.dump(st.session_state.users_db, f)
    except Exception as e:
        st.error(f"Error saving database: {e}")

def user_exists(username):
    """Check if a user exists in the database"""
    return username in st.session_state.users_db

def hash_password(password):
    """Hash a password for secure storage"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, email, password):
    """Create a new user in the database"""
    if user_exists(username):
        return False
    
    # Hash the password
    hashed_password = hash_password(password)
    
    # Create user record
    st.session_state.users_db[username] = {
        'email': email,
        'password': hashed_password,
        'join_date': datetime.now().strftime("%Y-%m-%d"),
        'demo_balance': 1000000.0,  # Start with $1,000,000 in demo account for stocks
        'demo_crypto_balance': 1000000.0,  # Start with $1,000,000 in demo account for crypto
        'portfolio': {},  # Stock portfolio
        'crypto_portfolio': {},  # Cryptocurrency portfolio
        'trading_history': [],  # Stock trading history
        'crypto_trading_history': [],  # Cryptocurrency trading history
        'prediction_history': []
    }
    
    # Save to file
    save_db()
    return True

def verify_user(username, password):
    """Verify user credentials"""
    if not user_exists(username):
        return False
    
    # Hash the provided password and compare
    hashed_password = hash_password(password)
    return st.session_state.users_db[username]['password'] == hashed_password

def get_user_data(username):
    """Get user data from the database"""
    if not user_exists(username):
        return None
    
    return st.session_state.users_db[username]

def update_user_data(username, data):
    """Update user data in the database"""
    if not user_exists(username):
        return False
    
    st.session_state.users_db[username].update(data)
    save_db()
    return True

def record_trade(username, symbol, company_name, quantity, price, total_amount, order_type, timestamp, auto_trade=False):
    """Record a trade in the user's trading history and update portfolio"""
    if not user_exists(username):
        return False
    
    user_data = st.session_state.users_db[username]
    
    # Record the trade
    trade = {
        'symbol': symbol,
        'company_name': company_name,
        'quantity': quantity,
        'price': price,
        'total_amount': total_amount,
        'order_type': order_type,
        'timestamp': timestamp,
        'auto_trade': auto_trade
    }
    
    user_data['trading_history'].append(trade)
    
    # Update user's balance
    if order_type == 'buy':
        # Check if enough balance
        if user_data.get('demo_balance', 0) < total_amount:
            return False
        
        user_data['demo_balance'] -= total_amount
        
        # Update portfolio
        if symbol not in user_data['portfolio']:
            user_data['portfolio'][symbol] = {
                'company_name': company_name,
                'quantity': 0,
                'avg_price': 0
            }
        
        # Calculate new average price
        current = user_data['portfolio'][symbol]
        current_value = current['quantity'] * current['avg_price']
        new_value = current_value + total_amount
        new_quantity = current['quantity'] + quantity
        
        current['quantity'] = new_quantity
        current['avg_price'] = new_value / new_quantity if new_quantity > 0 else 0
    
    elif order_type == 'sell':
        # Check if enough shares
        if symbol not in user_data['portfolio'] or user_data['portfolio'][symbol]['quantity'] < quantity:
            return False
        
        user_data['demo_balance'] += total_amount
        
        # Update portfolio
        current = user_data['portfolio'][symbol]
        current['quantity'] -= quantity
        
        # Remove stock from portfolio if quantity is 0
        if current['quantity'] == 0:
            del user_data['portfolio'][symbol]
    
    # Update portfolio history
    update_portfolio_history(username)
    
    # Save to file
    save_db()
    return True

def get_trading_history_as_df(username):
    """Get the user's trading history as a pandas DataFrame"""
    if not user_exists(username):
        return None
    
    history = st.session_state.users_db[username]['trading_history']
    if not history:
        return pd.DataFrame()
    
    return pd.DataFrame(history)

def calculate_portfolio_value(portfolio):
    """Calculate the current value of the stock portfolio"""
    if not portfolio:
        return 0
    
    portfolio_value = 0
    for symbol, details in portfolio.items():
        try:
            # Get current price
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Add to portfolio value
            portfolio_value += details['quantity'] * current_price
        except:
            # Use average price if can't get current price
            portfolio_value += details['quantity'] * details.get('avg_price', 0)
    
    return portfolio_value

def calculate_crypto_portfolio_value(crypto_portfolio):
    """Calculate the current value of the cryptocurrency portfolio"""
    if not crypto_portfolio:
        return 0
    
    portfolio_value = 0
    for symbol, details in crypto_portfolio.items():
        try:
            # Get current price
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Add to portfolio value
            portfolio_value += details['quantity'] * current_price
        except:
            # Use average price if can't get current price
            portfolio_value += details['quantity'] * details.get('avg_price', 0)
    
    return portfolio_value

def update_portfolio_history(username):
    """Update the user's portfolio history with current value"""
    if not user_exists(username):
        return False
    
    user_data = st.session_state.users_db[username]
    
    # Calculate portfolio value
    portfolio_value = calculate_portfolio_value(user_data['portfolio'])
    
    # Calculate crypto portfolio value if it exists
    crypto_portfolio_value = calculate_crypto_portfolio_value(user_data.get('crypto_portfolio', {}))
    
    # Get demo balance
    demo_balance = user_data.get('demo_balance', 0)
    
    # Add entry to portfolio history
    if 'portfolio_history' not in user_data:
        user_data['portfolio_history'] = []
    
    user_data['portfolio_history'].append({
        'date': datetime.now().strftime("%Y-%m-%d"),
        'cash': demo_balance,
        'portfolio_value': portfolio_value + crypto_portfolio_value,
        'total_value': demo_balance + portfolio_value + crypto_portfolio_value
    })
    
    # Limit history to last 365 entries
    if len(user_data['portfolio_history']) > 365:
        user_data['portfolio_history'] = user_data['portfolio_history'][-365:]
    
    # Save to file
    save_db()
    return True

def get_portfolio_history(username):
    """Get the user's portfolio history"""
    if not user_exists(username):
        return None
    
    user_data = st.session_state.users_db[username]
    
    if 'portfolio_history' not in user_data:
        user_data['portfolio_history'] = []
    
    return user_data['portfolio_history']

def reset_demo_account(username):
    """Reset the user's demo account to initial state"""
    if not user_exists(username):
        return False
    
    user_data = st.session_state.users_db[username]
    
    # Reset balance and portfolio
    user_data['demo_balance'] = 1000000.0
    user_data['demo_crypto_balance'] = 1000000.0
    user_data['portfolio'] = {}
    user_data['crypto_portfolio'] = {}
    
    # Add a note in trading history
    user_data['trading_history'].append({
        'symbol': 'SYSTEM',
        'company_name': 'System',
        'quantity': 0,
        'price': 0,
        'total_amount': 0,
        'order_type': 'reset',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Add a note in crypto trading history if it exists
    if 'crypto_trading_history' in user_data:
        user_data['crypto_trading_history'].append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'symbol': 'SYSTEM',
            'name': 'System',
            'quantity': 0,
            'price': 0,
            'total_amount': 0,
            'order_type': 'reset',
            'auto_trade': False
        })
    
    # Reset portfolio history
    user_data['portfolio_history'] = [{
        'date': datetime.now().strftime("%Y-%m-%d"),
        'cash': 1000000.0,
        'portfolio_value': 0.0,
        'total_value': 1000000.0
    }]
    
    # Save to file
    save_db()
    return True

def save_prediction(username, prediction_data):
    """Save a stock or cryptocurrency prediction to the user's prediction history"""
    if not user_exists(username):
        return False
    
    user_data = st.session_state.users_db[username]
    
    if 'prediction_history' not in user_data:
        user_data['prediction_history'] = []
    
    # Check if it's a cryptocurrency prediction
    if 'symbol' in prediction_data:
        symbol = prediction_data['symbol']
        if symbol.endswith('-USD') or symbol.endswith('-USDT'):
            prediction_data['is_crypto'] = True
    
    user_data['prediction_history'].append(prediction_data)
    
    # Limit history to last 100 predictions
    if len(user_data['prediction_history']) > 100:
        user_data['prediction_history'] = user_data['prediction_history'][-100:]
    
    # Save to file
    save_db()
    return True

def get_prediction_history(username, crypto_only=False):
    """Get the user's prediction history, optionally filtered to crypto only"""
    if not user_exists(username):
        return None
    
    user_data = st.session_state.users_db[username]
    
    if 'prediction_history' not in user_data:
        user_data['prediction_history'] = []
    
    if not crypto_only:
        return user_data['prediction_history']
    else:
        # Filter to show only cryptocurrency predictions
        return [
            prediction for prediction in user_data['prediction_history']
            if ('symbol' in prediction and 
                (prediction['symbol'].endswith('-USD') or 
                 prediction['symbol'].endswith('-USDT') or
                 prediction.get('is_crypto', False)))
        ]
