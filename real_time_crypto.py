import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import json

# Import custom modules
import data_fetcher
import database
import utils

def show_real_time_crypto_portfolio():
    """Display real-time cryptocurrency portfolio with live updates"""
    st.markdown("""
    <div class="dashboard-header">
        <h1>Real-Time Cryptocurrency Portfolio</h1>
        <p>Live tracking of your cryptocurrency investments with real-time P/L updates</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user is logged in
    if 'username' not in st.session_state or not st.session_state.username:
        st.warning("Please log in to view your portfolio.")
        return
    
    # Create main content container
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    
    # Create placeholder for the portfolio summary metrics
    summary_metrics = st.empty()
    
    # Create placeholder for the portfolio chart
    portfolio_chart = st.empty()
    
    # Create placeholder for the portfolio table
    portfolio_table = st.empty()
    
    # Add controls for refresh rate
    st.markdown("<h4>Real-Time Settings</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        auto_refresh = st.toggle("Auto-refresh", value=True)
    
    with col2:
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=10, step=5)
        else:
            refresh_interval = 30  # Default value
            st.button("Refresh Now", on_click=lambda: None)
    
    # Function to update portfolio data
    def update_portfolio_data():
        """Fetch and format the latest portfolio data"""
        # Get user data
        user_data = database.get_user_data(st.session_state.username)
        
        if user_data is None:
            return None, None, None, None, 0
        
        # Ensure crypto_portfolio exists in user data
        crypto_portfolio = user_data.get('crypto_portfolio', {})
        
        # Calculate total portfolio value
        total_crypto_value = 0
        portfolio_data = []
        crypto_assets_data = {}
        
        # Get timestamp
        current_time = datetime.now().strftime('%H:%M:%S')
        
        for symbol, data in crypto_portfolio.items():
            # Get current price
            crypto_data = data_fetcher.get_crypto_data(symbol.split('-')[0], period="1d")
            
            if crypto_data is not None and not crypto_data.empty:
                current_price = crypto_data['Close'].iloc[-1]
                quantity = data.get('quantity', 0)
                avg_cost = data.get('avg_price', 0)
                
                current_value = quantity * current_price
                cost_basis = quantity * avg_cost
                profit_loss = current_value - cost_basis
                profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
                
                total_crypto_value += current_value
                
                # Get crypto name from symbol
                crypto_name = next((name for name, sym in data_fetcher.MAJOR_CRYPTOCURRENCIES.items() if sym == symbol), symbol)
                
                portfolio_data.append({
                    'Name': crypto_name,
                    'Symbol': symbol.split('-')[0],
                    'Quantity': quantity,
                    'Average Cost': avg_cost,
                    'Current Price': current_price,
                    'Current Value': current_value,
                    'Profit/Loss': profit_loss,
                    'Profit/Loss %': profit_loss_pct,
                    'Time': current_time
                })
                
                # Store data for the chart
                if symbol not in crypto_assets_data:
                    crypto_assets_data[symbol] = []
                
                crypto_assets_data[symbol].append({
                    'time': current_time,
                    'value': current_value,
                    'pl': profit_loss,
                    'pl_pct': profit_loss_pct
                })
        
        # Get cash balance
        crypto_cash = st.session_state.get('crypto_balance', 1000000.0)
        
        return portfolio_data, total_crypto_value, crypto_cash, crypto_assets_data, len(crypto_portfolio)
    
    # Function to display portfolio metrics
    def display_portfolio_metrics(total_value, cash_balance):
        """Display portfolio summary metrics"""
        col1, col2, col3 = summary_metrics.columns(3)
        
        # Total portfolio value (cash + crypto)
        total_portfolio = cash_balance + total_value
        
        col1.metric("Total Portfolio Value", f"${total_portfolio:.2f}")
        col2.metric("Cash Balance", f"${cash_balance:.2f}")
        col3.metric("Crypto Holdings Value", f"${total_value:.2f}")
    
    # Function to display portfolio chart
    def display_portfolio_chart(crypto_assets_data, portfolio_size):
        """Display interactive chart of portfolio value and P/L over time"""
        if portfolio_size == 0:
            portfolio_chart.info("No cryptocurrency holdings to display. Start trading to see your portfolio performance.")
            return
        
        # Prepare data for the chart
        all_assets_data = []
        
        for symbol, data_points in crypto_assets_data.items():
            for point in data_points[-10:]:  # Show only last 10 data points to avoid cluttering
                name = next((name for name, sym in data_fetcher.MAJOR_CRYPTOCURRENCIES.items() if sym == symbol), symbol)
                all_assets_data.append({
                    'Asset': name,
                    'Time': point['time'],
                    'Value': point['value'],
                    'P/L': point['pl'],
                    'P/L %': point['pl_pct']
                })
        
        if not all_assets_data:
            return
            
        df = pd.DataFrame(all_assets_data)
        
        # Create a plot with two y-axes
        fig = go.Figure()
        
        # Plot each asset separately
        for asset in df['Asset'].unique():
            asset_df = df[df['Asset'] == asset]
            fig.add_trace(go.Scatter(
                x=asset_df['Time'], 
                y=asset_df['Value'],
                name=f"{asset} Value",
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Real-Time Cryptocurrency Asset Values",
            xaxis_title="Time",
            yaxis_title="Value (USD)",
            hovermode="x unified",
            height=400,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        portfolio_chart.plotly_chart(fig, use_container_width=True)
        
        # Create a second chart for P/L percentage
        fig2 = go.Figure()
        
        for asset in df['Asset'].unique():
            asset_df = df[df['Asset'] == asset]
            fig2.add_trace(go.Scatter(
                x=asset_df['Time'], 
                y=asset_df['P/L %'],
                name=f"{asset} P/L %",
                line=dict(width=2)
            ))
        
        fig2.update_layout(
            title="Real-Time Profit/Loss Percentage",
            xaxis_title="Time",
            yaxis_title="P/L %",
            hovermode="x unified",
            height=350,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Function to display portfolio table
    def display_portfolio_table(portfolio_data):
        """Display formatted table of cryptocurrency holdings"""
        if not portfolio_data:
            portfolio_table.info("No cryptocurrency holdings to display. Start trading to build your portfolio!")
            return
        
        df = pd.DataFrame(portfolio_data)
        
        # Format the data for display
        def highlight_profit_loss(val):
            if val > 0:
                return f'<span style="color:#4CAF50">+{val:.2f}%</span>'
            elif val < 0:
                return f'<span style="color:#FF5252">{val:.2f}%</span>'
            else:
                return f'<span style="color:#FFFFFF">{val:.2f}%</span>'
        
        # Format numeric columns
        # Ensure no empty strings in numeric columns
        for col in ['Quantity', 'Average Cost', 'Current Price', 'Current Value', 'Profit/Loss', 'Profit/Loss %']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Now format the numeric columns
        df['Quantity'] = df['Quantity'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "0.0000")
        df['Average Cost'] = df['Average Cost'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
        df['Current Price'] = df['Current Price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
        df['Current Value'] = df['Current Value'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
        df['Profit/Loss'] = df['Profit/Loss'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "$0.00")
        df['Profit/Loss %'] = df['Profit/Loss %'].apply(highlight_profit_loss)
        
        # Remove the Time column for the table display
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)
        
        # Display the formatted dataframe
        portfolio_table.markdown("<h4>Your Cryptocurrency Holdings</h4>", unsafe_allow_html=True)
        portfolio_table.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Create a container for real-time data
    data_container = st.empty()
    
    # Store data history for charts
    if 'crypto_assets_history' not in st.session_state:
        st.session_state.crypto_assets_history = {}
    
    # First update to initialize
    portfolio_data, total_value, cash_balance, assets_data, portfolio_size = update_portfolio_data()
    
    # Update session state with new data
    for symbol, data in assets_data.items() if assets_data else {}.items():
        if symbol not in st.session_state.crypto_assets_history:
            st.session_state.crypto_assets_history[symbol] = []
        st.session_state.crypto_assets_history[symbol].extend(data)
    
    # Display data using the helper functions
    display_portfolio_metrics(total_value if total_value else 0, cash_balance if cash_balance else 1000000.0)
    display_portfolio_chart(st.session_state.crypto_assets_history, portfolio_size)
    display_portfolio_table(portfolio_data)
    
    # Auto-refresh logic
    if auto_refresh:
        placeholder = st.empty()
        
        with placeholder.container():
            while auto_refresh:
                # Display a countdown timer
                for remaining in range(refresh_interval, 0, -1):
                    progress_text = f"Next update in {remaining} seconds..."
                    st.markdown(f"""
                    <div style="background-color: rgba(70, 130, 180, 0.1); padding: 5px 10px; border-radius: 5px; text-align: center;">
                        {progress_text}
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    if not auto_refresh:
                        break
                
                if not auto_refresh:
                    break
                
                st.markdown("Updating data...")
                
                # Update portfolio data
                portfolio_data, total_value, cash_balance, assets_data, portfolio_size = update_portfolio_data()
                
                # Update session state with new data
                for symbol, data in assets_data.items() if assets_data else {}.items():
                    if symbol not in st.session_state.crypto_assets_history:
                        st.session_state.crypto_assets_history[symbol] = []
                    st.session_state.crypto_assets_history[symbol].extend(data)
                
                # Trim history to keep only the most recent points (to avoid memory issues)
                for symbol in st.session_state.crypto_assets_history:
                    if len(st.session_state.crypto_assets_history[symbol]) > 100:
                        st.session_state.crypto_assets_history[symbol] = st.session_state.crypto_assets_history[symbol][-100:]
                
                # Refresh the displays
                display_portfolio_metrics(total_value if total_value else 0, cash_balance if cash_balance else 1000000.0)
                display_portfolio_chart(st.session_state.crypto_assets_history, portfolio_size)
                display_portfolio_table(portfolio_data)
                
                st.markdown("Data updated!")
                
    st.markdown('</div>', unsafe_allow_html=True)

# Function to fetch real-time cryptocurrency market data
def show_real_time_crypto_market():
    """Display real-time cryptocurrency market overview"""
    st.markdown("""
    <div class="dashboard-header">
        <h1>Real-Time Cryptocurrency Market</h1>
        <p>Live market data and price movements for major cryptocurrencies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main content container
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    
    # Create columns for the market metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Top Cryptocurrencies</h3>", unsafe_allow_html=True)
        
        # Create a placeholder for the top cryptocurrencies table
        top_crypto_table = st.empty()
    
    with col2:
        st.markdown("<h3>Market Metrics</h3>", unsafe_allow_html=True)
        
        # Create a placeholder for market metrics
        market_metrics_placeholder = st.empty()
    
    # Create a placeholder for price charts
    price_chart_placeholder = st.empty()
    
    # Controls for auto-refresh
    st.markdown("<h4>Real-Time Settings</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        auto_refresh = st.toggle("Auto-refresh", value=True)
    
    with col2:
        if auto_refresh:
            refresh_interval = st.slider("Refresh interval (seconds)", min_value=5, max_value=60, value=10, step=5)
        else:
            refresh_interval = 30  # Default value
            st.button("Refresh Now", on_click=lambda: None)
    
    # Function to update market data
    def update_market_data():
        """Fetch and update the cryptocurrency market data"""
        # Get top cryptocurrencies
        top_cryptos = data_fetcher.get_top_cryptocurrencies(limit=10)
        
        # Get market overview
        market_overview = data_fetcher.get_crypto_market_overview()
        
        return top_cryptos, market_overview
    
    # Function to display top cryptocurrencies
    def display_top_cryptos(top_cryptos):
        """Display a formatted table of top cryptocurrencies"""
        if top_cryptos is None or len(top_cryptos) == 0:
            top_crypto_table.warning("Unable to fetch cryptocurrency data. Please try again later.")
            return
        
        # Convert list to DataFrame if it's a list
        df = pd.DataFrame(top_cryptos) if isinstance(top_cryptos, list) else top_cryptos.copy()
        
        # Function to highlight change
        def highlight_change(val):
            if val > 0:
                return f'<span style="color:#4CAF50">+{val:.2f}%</span>'
            elif val < 0:
                return f'<span style="color:#FF5252">{val:.2f}%</span>'
            else:
                return f'<span style="color:#FFFFFF">{val:.2f}%</span>'
        
        # Format columns
        if 'Price' in df.columns:
            df['Price'] = df['Price'].apply(lambda x: f"${x:.2f}" if x > 1 else f"${x:.6f}")
        
        if 'Market Cap' in df.columns:
            df['Market Cap'] = df['Market Cap'].apply(utils.format_large_number)
        
        if 'Volume (24h)' in df.columns:
            df['Volume (24h)'] = df['Volume (24h)'].apply(utils.format_large_number)
        
        if 'Change (24h)' in df.columns:
            df['Change (24h)'] = df['Change (24h)'].apply(highlight_change)
        
        # Display the table
        top_crypto_table.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Function to display market metrics
    def display_market_metrics(market_overview):
        """Display cryptocurrency market metrics"""
        if market_overview is None:
            market_metrics_placeholder.warning("Unable to fetch market overview. Please try again later.")
            return
        
        col1, col2 = market_metrics_placeholder.columns(2)
        
        # Display metrics
        col1.metric("Total Market Cap", utils.format_large_number(market_overview.get('total_market_cap', 0)))
        col2.metric("24h Trading Volume", utils.format_large_number(market_overview.get('total_volume_24h', 0)))
        
        col1.metric("BTC Dominance", f"{market_overview.get('btc_dominance', 0):.2f}%")
        col2.metric("Market Change (24h)", utils.format_percentage(market_overview.get('market_change_24h', 0)))
        
        # Additional metrics
        col1.metric("Active Cryptocurrencies", f"{market_overview.get('active_cryptocurrencies', 0):,}")
        col2.metric("Active Exchanges", f"{market_overview.get('active_exchanges', 0):,}")
    
    # Function to display price charts
    def display_price_charts(top_cryptos):
        """Display interactive price charts for selected cryptocurrencies"""
        if top_cryptos is None or (isinstance(top_cryptos, list) and len(top_cryptos) == 0):
            return
        
        # Convert list to DataFrame if it's a list
        df = pd.DataFrame(top_cryptos) if isinstance(top_cryptos, list) else top_cryptos
        
        if df.empty:
            return
            
        st.markdown("<h3>Live Price Charts</h3>", unsafe_allow_html=True)
        
        # Select cryptocurrencies to display
        selected_cryptos = st.multiselect(
            "Select cryptocurrencies to display",
            options=df['Name'].tolist(),
            default=df['Name'].tolist()[:3] if len(df) >= 3 else df['Name'].tolist()  # Default to top 3
        )
        
        if not selected_cryptos:
            st.info("Please select at least one cryptocurrency to display charts.")
            return
        
        # Get symbols for selected cryptos
        symbols = []
        for name in selected_cryptos:
            # Handle both DataFrame and list
            if isinstance(top_cryptos, pd.DataFrame):
                symbol_row = top_cryptos[top_cryptos['Name'] == name]
                if not symbol_row.empty:
                    symbol = symbol_row['Symbol'].iloc[0]
                    symbols.append(symbol)
            else:  # It's a list
                for crypto in top_cryptos:
                    if crypto['Name'] == name:
                        symbols.append(crypto['Symbol'])
                        break
        
        # Create charts
        for symbol in symbols:
            # Get crypto data
            crypto_data = data_fetcher.get_crypto_data(symbol, period="1d")
            
            if crypto_data is not None and not crypto_data.empty:
                # Get name - handle both DataFrame and list
                if isinstance(top_cryptos, pd.DataFrame):
                    name_row = top_cryptos[top_cryptos['Symbol'] == symbol]
                    name = name_row['Name'].iloc[0] if not name_row.empty else symbol
                else:  # It's a list
                    name = next((crypto['Name'] for crypto in top_cryptos if crypto['Symbol'] == symbol), symbol)
                
                # Create candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=crypto_data.index,
                    open=crypto_data['Open'],
                    high=crypto_data['High'],
                    low=crypto_data['Low'],
                    close=crypto_data['Close'],
                    name=name
                )])
                
                # Calculate current price and 24h change
                current_price = crypto_data['Close'].iloc[-1]
                prev_price = crypto_data['Close'].iloc[0]
                price_change = ((current_price - prev_price) / prev_price) * 100
                
                # Add chart title with current price and change
                fig.update_layout(
                    title=f"{name} ({symbol}) - ${current_price:.2f} ({'+' if price_change >= 0 else ''}{price_change:.2f}%)",
                    xaxis_title="Time",
                    yaxis_title="Price (USD)",
                    height=400,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.1)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Initial data fetch
    top_cryptos, market_overview = update_market_data()
    
    # Display the data
    display_top_cryptos(top_cryptos)
    display_market_metrics(market_overview)
    display_price_charts(top_cryptos)
    
    # Auto-refresh logic
    if auto_refresh:
        placeholder = st.empty()
        
        with placeholder.container():
            while auto_refresh:
                # Display a countdown timer
                for remaining in range(refresh_interval, 0, -1):
                    progress_text = f"Next update in {remaining} seconds..."
                    st.markdown(f"""
                    <div style="background-color: rgba(70, 130, 180, 0.1); padding: 5px 10px; border-radius: 5px; text-align: center;">
                        {progress_text}
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    if not auto_refresh:
                        break
                
                if not auto_refresh:
                    break
                
                st.markdown("Updating market data...")
                
                # Update market data
                top_cryptos, market_overview = update_market_data()
                
                # Refresh the displays
                display_top_cryptos(top_cryptos)
                display_market_metrics(market_overview)
                
                st.markdown("Market data updated!")
    
    st.markdown('</div>', unsafe_allow_html=True)