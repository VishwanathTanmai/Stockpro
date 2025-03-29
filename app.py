import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# Import modules
import auth
import dashboard
import portfolio
import trading
import prediction
import data_fetcher
import database
import auto_trading
import stock_comparison
import stock_screener
import price_alerts
import real_time_monitor
import cryptocurrency
import real_time_crypto


# Page configuration
st.set_page_config(
    page_title="StockPro - Stock Market Application",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# No custom CSS - using default Streamlit UI

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'
if 'crypto_balance' not in st.session_state:
    st.session_state.crypto_balance = 1000000.0

# Initialize the database when the app starts
database.initialize_db()

# App header with standard styling
st.title("📈 StockPro")
st.subheader("Advanced Stock Market Analysis & Trading Platform")

# Handle authentication first
if not st.session_state.logged_in:
    auth.show_auth_page()
else:
    # Display sidebar navigation with standard styling
    with st.sidebar:
        # User welcome section
        st.write(f"👤 Welcome, {st.session_state.username}!")
        st.write(f"Last login: {datetime.now().strftime('%b %d, %Y')}")
        st.divider()
        
        # Navigation options
        nav_options = {
            "dashboard": {"icon": "📊", "title": "Dashboard", "category": "main"},
            "portfolio": {"icon": "📁", "title": "Portfolio", "category": "main"},
            "real_time_monitor": {"icon": "⚡", "title": "Real-Time Monitor", "category": "main"},
            "crypto_dashboard": {"icon": "₿", "title": "Crypto Market", "category": "crypto"},
            "real_time_crypto_portfolio": {"icon": "📊", "title": "Real-Time Crypto Portfolio", "category": "crypto"},
            "real_time_crypto_market": {"icon": "📈", "title": "Real-Time Crypto Market", "category": "crypto"},
            "trading": {"icon": "💹", "title": "Stock Trading", "category": "trading"},
            "crypto_trading": {"icon": "🪙", "title": "Crypto Trading", "category": "trading"},
            "auto_trading": {"icon": "🤖", "title": "Auto Trading", "category": "trading"},
            "prediction": {"icon": "🔮", "title": "Stock Predictions", "category": "analysis"},
            "crypto_prediction": {"icon": "📈", "title": "Crypto Predictions", "category": "analysis"},
            "stock_comparison": {"icon": "🔍", "title": "Stock Comparison", "category": "analysis"},
            "crypto_comparison": {"icon": "⚖️", "title": "Crypto Comparison", "category": "analysis"},
            "stock_screener": {"icon": "🔎", "title": "Stock Screener", "category": "analysis"},
            "price_alerts": {"icon": "🔔", "title": "Price Alerts", "category": "alerts"},
            "profile": {"icon": "👤", "title": "Profile", "category": "account"}
        }
        
        # Main section
        st.subheader("🏠 Main")
        for page, info in nav_options.items():
            if info["category"] == "main":
                if st.button(f"{info['icon']} {info['title']}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
        
        # Cryptocurrency section
        st.subheader("₿ Cryptocurrency")
        for page, info in nav_options.items():
            if info["category"] == "crypto":
                if st.button(f"{info['icon']} {info['title']}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
        
        # Trading section
        st.subheader("💰 Trading")
        for page, info in nav_options.items():
            if info["category"] == "trading":
                if st.button(f"{info['icon']} {info['title']}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
        
        # Analysis section
        st.subheader("📊 Analysis")
        for page, info in nav_options.items():
            if info["category"] == "analysis":
                if st.button(f"{info['icon']} {info['title']}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
        
        # Alerts & Account section
        st.subheader("⚙️ Settings & Alerts")
        for page, info in nav_options.items():
            if info["category"] in ["alerts", "account"]:
                if st.button(f"{info['icon']} {info['title']}", key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
                
        # Logout button
        st.divider()
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            auth.logout()
            st.rerun()
        
        # App version info at the bottom
        st.caption("StockPro v2.0.0 - Advanced Edition")
    
    # Display the selected page
    if st.session_state.current_page == 'dashboard':
        dashboard.show_dashboard()
    
    elif st.session_state.current_page == 'portfolio':
        portfolio.show_portfolio()
    
    elif st.session_state.current_page == 'real_time_monitor':
        real_time_monitor.show_real_time_monitor()
        real_time_monitor.create_trend_analysis()
        real_time_monitor.show_heatmap()
        real_time_monitor.show_economic_indicators()
    
    # Cryptocurrency pages
    elif st.session_state.current_page == 'crypto_dashboard':
        cryptocurrency.show_crypto_dashboard()
    
    elif st.session_state.current_page == 'crypto_trading':
        cryptocurrency.crypto_trading()
    
    elif st.session_state.current_page == 'crypto_comparison':
        cryptocurrency.crypto_comparison()
    
    elif st.session_state.current_page == 'crypto_prediction':
        cryptocurrency.crypto_prediction()
        
    elif st.session_state.current_page == 'real_time_crypto_portfolio':
        real_time_crypto.show_real_time_crypto_portfolio()
        
    elif st.session_state.current_page == 'real_time_crypto_market':
        real_time_crypto.show_real_time_crypto_market()
    
    # Stock trading pages
    elif st.session_state.current_page == 'trading':
        trading.show_trading_page()
    
    elif st.session_state.current_page == 'auto_trading':
        auto_trading.show_auto_trading_page()
    
    elif st.session_state.current_page == 'prediction':
        prediction.show_prediction_page()
    
    elif st.session_state.current_page == 'stock_comparison':
        stock_comparison.show_comparison_page()
    
    elif st.session_state.current_page == 'stock_screener':
        stock_screener.show_screener_page()
    
    elif st.session_state.current_page == 'price_alerts':
        price_alerts.show_alerts_page()
    
    elif st.session_state.current_page == 'profile':
        auth.show_profile()

# Footer with standard styling
st.divider()
current_year = datetime.now().year

col1, col2 = st.columns(2)
with col1:
    st.write("📈 StockPro")
    st.write("Advanced market analytics and trading tools for investors")
with col2:
    st.write(f"© {current_year} StockPro | Powered by Yahoo Finance")
    st.caption("Real-time data • AI predictions • Market Heatmaps • Economic Indicators")
