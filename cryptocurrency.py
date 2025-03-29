import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

import data_fetcher
import utils
import database

def show_crypto_dashboard():
    """Display the cryptocurrency dashboard with market overview"""
    st.markdown("""
    <div class="dashboard-header">
        <h1>Cryptocurrency Market Dashboard</h1>
        <p>Real-time cryptocurrency market data and performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display market overview in an enhanced container
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    
    # Get market overview data
    market_data = data_fetcher.get_crypto_market_overview()
    
    if market_data:
        # Create columns for market metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "BTC Price", 
                f"${market_data['btc_price']:,.2f}", 
                f"{market_data['btc_change']:.2f}%"
            )
            
        with col2:
            st.metric(
                "ETH Price", 
                f"${market_data['eth_price']:,.2f}", 
                f"{market_data['eth_change']:.2f}%"
            )
            
        with col3:
            st.metric(
                "Market Sentiment", 
                f"{market_data['sentiment']:.1f}/100", 
                f"{market_data['sentiment_change']:.2f}%"
            )
            
        with col4:
            st.metric(
                "BTC Dominance", 
                f"{market_data['btc_dominance']:.2f}%", 
                None
            )
        
        # Market visualization 
        st.markdown("<h3>Cryptocurrency Market Sentiment</h3>", unsafe_allow_html=True)
        
        # Create a sentiment gauge visualization
        sentiment = market_data['sentiment']
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1E88E5"},
                'steps': [
                    {'range': [0, 33], 'color': "#FF5252"},  # Fear
                    {'range': [33, 67], 'color': "#FFC107"}, # Neutral
                    {'range': [67, 100], 'color': "#4CAF50"} # Greed
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.8,
                    'value': sentiment
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#FFFFFF")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Unable to fetch cryptocurrency market data. Please try again later.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display top cryptocurrencies with performance
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    st.markdown("<h2>Top Cryptocurrencies</h2>", unsafe_allow_html=True)
    
    crypto_data = data_fetcher.get_top_cryptocurrencies(limit=10)
    
    if crypto_data:
        df = pd.DataFrame(crypto_data)
        
        # Function to apply color formatting to percentage changes
        def highlight_change(val):
            if val > 0:
                return f'<span style="color:#4CAF50">+{val:.2f}%</span>'
            elif val < 0:
                return f'<span style="color:#FF5252">{val:.2f}%</span>'
            else:
                return f'<span style="color:#FFFFFF">{val:.2f}%</span>'
        
        # Format numeric data
        df['Price'] = df['Price'].apply(lambda x: f"${x:,.2f}")
        df['Market Cap'] = df['Market Cap'].apply(lambda x: utils.format_large_number(x) if pd.notnull(x) else "N/A")
        df['Volume (24h)'] = df['Volume (24h)'].apply(lambda x: utils.format_large_number(x) if pd.notnull(x) else "N/A")
        
        # Apply highlighting to change percentages
        df['Change %'] = df['Change %'].apply(highlight_change)
        
        # Display as HTML for better formatting
        styled_df = df[['Name', 'Symbol', 'Price', 'Change %', 'Market Cap', 'Volume (24h)']]
        st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        
    else:
        st.error("Unable to fetch cryptocurrency data. Please try again later.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display Bitcoin and Ethereum charts
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    st.markdown("<h2>Price Charts</h2>", unsafe_allow_html=True)
    
    # Chart period selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        crypto_symbols = ["BTC", "ETH", "SOL", "XRP", "BNB"]
        selected_crypto = st.selectbox("Select Cryptocurrency", crypto_symbols)
        
    with col2:
        period_options = {
            "1d": "1 Day",
            "5d": "5 Days",
            "1mo": "1 Month",
            "3mo": "3 Months",
            "6mo": "6 Months",
            "1y": "1 Year",
            "ytd": "Year to Date",
            "max": "Max"
        }
        selected_period = st.selectbox("Select Period", list(period_options.keys()), format_func=lambda x: period_options[x])
    
    # Get cryptocurrency data
    crypto_data = data_fetcher.get_crypto_data(selected_crypto, period=selected_period)
    
    if crypto_data is not None and not crypto_data.empty:
        # Create price chart with plotly
        fig = utils.create_candlestick_chart(crypto_data, title=f"{selected_crypto} Price Chart")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add some key stats below the chart
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate stats
        current_price = crypto_data['Close'].iloc[-1]
        price_change = (crypto_data['Close'].iloc[-1] - crypto_data['Close'].iloc[0]) / crypto_data['Close'].iloc[0] * 100
        high_price = crypto_data['High'].max()
        low_price = crypto_data['Low'].min()
        
        col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f}%")
        col2.metric("Period High", f"${high_price:.2f}")
        col3.metric("Period Low", f"${low_price:.2f}")
        col4.metric("Trading Volume", utils.format_large_number(crypto_data['Volume'].mean()))
    
    else:
        st.error(f"Unable to fetch data for {selected_crypto}. Please try a different cryptocurrency or time period.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display cryptocurrency news
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    st.markdown("<h2>Cryptocurrency News</h2>", unsafe_allow_html=True)
    
    news_items = data_fetcher.get_crypto_news()
    
    if news_items:
        for i, news in enumerate(news_items):
            st.markdown(f"""
            <div class="news-item">
                <h3 class="news-title">{news['title']}</h3>
                <div class="news-meta">
                    <span class="news-source">{news['source']}</span> • 
                    <span class="news-date">{news['published']}</span>
                </div>
                <p class="news-summary">{news['summary']}</p>
                <a href="{news['link']}" target="_blank" class="news-link">Read More →</a>
            </div>
            """, unsafe_allow_html=True)
            
            if i < len(news_items) - 1:
                st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.warning("No cryptocurrency news available at the moment.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def crypto_trading():
    """Show cryptocurrency trading interface"""
    st.markdown("""
    <div class="dashboard-header">
        <h1>Cryptocurrency Trading</h1>
        <p>Buy and sell cryptocurrencies with your demo account</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display trading interface
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<h3>Place Order</h3>", unsafe_allow_html=True)
        
        # Cryptocurrency selection
        crypto_options = list(data_fetcher.MAJOR_CRYPTOCURRENCIES.items())
        selected_crypto = st.selectbox(
            "Select Cryptocurrency",
            options=[name for name, _ in crypto_options],
            index=0
        )
        
        # Get the actual ticker symbol
        selected_symbol = [symbol for name, symbol in crypto_options if name == selected_crypto][0]
        
        # Get current data for the selected crypto
        crypto_data = data_fetcher.get_crypto_data(selected_symbol.split('-')[0], period="1d")
        
        if crypto_data is not None and not crypto_data.empty:
            current_price = crypto_data['Close'].iloc[-1]
            
            # Display current price
            st.markdown(f"""
            <div class="current-price-display">
                <div class="price-label">Current Price:</div>
                <div class="price-value">${current_price:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Order type
            order_type = st.radio("Order Type", ["Buy", "Sell"], horizontal=True)
            
            # Quantity - validate as a positive number
            quantity = st.number_input("Quantity", min_value=0.0001, step=0.01, format="%.4f")
            
            # Calculate total
            total_amount = quantity * current_price
            
            # Display order summary
            st.markdown(f"""
            <div class="order-summary">
                <div class="summary-row">
                    <div class="summary-label">Price:</div>
                    <div class="summary-value">${current_price:.2f}</div>
                </div>
                <div class="summary-row">
                    <div class="summary-label">Quantity:</div>
                    <div class="summary-value">{quantity:.4f} {selected_symbol.split('-')[0]}</div>
                </div>
                <div class="summary-row total">
                    <div class="summary-label">Total:</div>
                    <div class="summary-value">${total_amount:.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if the user has a sufficient balance for buying
            if order_type == "Buy":
                # Get user data
                user_data = database.get_user_data(st.session_state.username)
                
                if user_data is not None:
                    demo_balance = user_data.get('demo_crypto_balance', 1000000.0)
                    
                    if total_amount > demo_balance:
                        st.warning(f"Insufficient funds. Your current crypto balance is ${demo_balance:.2f}")
                        st.button("Place Order", disabled=True)
                    else:
                        if st.button("Place Order"):
                            # Execute the buy order
                            execute_crypto_order(
                                st.session_state.username,
                                selected_symbol,
                                selected_crypto,
                                quantity,
                                current_price,
                                order_type
                            )
                            st.success(f"Order executed successfully: {order_type} {quantity:.4f} {selected_symbol.split('-')[0]} at ${current_price:.2f}")
                            st.rerun()
            else:  # Sell order
                user_data = database.get_user_data(st.session_state.username)
                
                if user_data is not None:
                    portfolio = user_data.get('crypto_portfolio', {})
                    
                    # Check if user owns the crypto
                    user_quantity = portfolio.get(selected_symbol, {}).get('quantity', 0)
                    
                    if quantity > user_quantity:
                        st.warning(f"Insufficient {selected_symbol.split('-')[0]}. Your current balance is {user_quantity:.4f}")
                        st.button("Place Order", disabled=True)
                    else:
                        if st.button("Place Order"):
                            # Execute the sell order
                            execute_crypto_order(
                                st.session_state.username,
                                selected_symbol,
                                selected_crypto,
                                quantity,
                                current_price,
                                order_type
                            )
                            st.success(f"Order executed successfully: {order_type} {quantity:.4f} {selected_symbol.split('-')[0]} at ${current_price:.2f}")
                            st.rerun()
        else:
            st.error(f"Unable to fetch data for {selected_crypto}. Please try a different cryptocurrency.")
    
    with col2:
        st.markdown("<h3>Portfolio Overview</h3>", unsafe_allow_html=True)
        
        # Get user data
        user_data = database.get_user_data(st.session_state.username)
        
        if user_data is not None:
            # Ensure demo_crypto_balance exists in user data
            if 'demo_crypto_balance' not in user_data:
                user_data['demo_crypto_balance'] = 1000000.0
                database.update_user_data(st.session_state.username, user_data)
                
            demo_balance = user_data.get('demo_crypto_balance', 1000000.0)
            crypto_portfolio = user_data.get('crypto_portfolio', {})
            
            # Calculate total portfolio value
            total_crypto_value = 0
            portfolio_data = []
            
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
                        'Profit/Loss %': profit_loss_pct
                    })
            
            # Display account balance
            col1, col2 = st.columns(2)
            # Use session state crypto_balance which is always up-to-date
            crypto_cash = st.session_state.get('crypto_balance', 1000000.0)
            col1.metric("Cash Balance", f"${crypto_cash:.2f}")
            col2.metric("Crypto Portfolio Value", f"${total_crypto_value:.2f}")
            
            # Display portfolio table if there are crypto holdings
            if portfolio_data:
                st.markdown("<h4>Your Cryptocurrency Holdings</h4>", unsafe_allow_html=True)
                
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
                df['Quantity'] = df['Quantity'].apply(lambda x: f"{x:.4f}")
                df['Average Cost'] = df['Average Cost'].apply(lambda x: f"${x:.2f}")
                df['Current Price'] = df['Current Price'].apply(lambda x: f"${x:.2f}")
                df['Current Value'] = df['Current Value'].apply(lambda x: f"${x:.2f}")
                df['Profit/Loss'] = df['Profit/Loss'].apply(lambda x: f"${x:.2f}")
                df['Profit/Loss %'] = df['Profit/Loss %'].apply(highlight_profit_loss)
                
                # Display the formatted dataframe
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("You don't have any cryptocurrency holdings. Start trading to build your portfolio!")
        else:
            st.error("Unable to fetch user data. Please try logging in again.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display cryptocurrency trading history
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    st.markdown("<h3>Trading History</h3>", unsafe_allow_html=True)
    
    # Get user's trading history
    user_data = database.get_user_data(st.session_state.username)
    
    if user_data is not None:
        crypto_trading_history = user_data.get('crypto_trading_history', [])
        
        if crypto_trading_history:
            # Convert to DataFrame for easier display
            df = pd.DataFrame(crypto_trading_history)
            
            # Sort by timestamp (most recent first)
            df = df.sort_values('timestamp', ascending=False)
            
            # Function to highlight order type
            def highlight_order_type(val):
                if val == "Buy":
                    return f'<span style="color:#4CAF50">Buy</span>'
                else:
                    return f'<span style="color:#FF5252">Sell</span>'
            
            # Format the data
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
            df['total_amount'] = df['total_amount'].apply(lambda x: f"${x:.2f}")
            df['order_type'] = df['order_type'].apply(highlight_order_type)
            
            # Rename columns for display
            df.columns = ['Date', 'Symbol', 'Name', 'Quantity', 'Price', 'Total', 'Type', 'Auto']
            
            # Select and reorder columns
            display_df = df[['Date', 'Type', 'Symbol', 'Name', 'Quantity', 'Price', 'Total']]
            
            # Display the table
            st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No trading history yet. Start trading to see your activity here.")
    else:
        st.error("Unable to fetch user data. Please try logging in again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def crypto_comparison():
    """Show cryptocurrency comparison tool"""
    st.markdown("""
    <div class="dashboard-header">
        <h1>Cryptocurrency Comparison</h1>
        <p>Compare performance metrics between cryptocurrencies</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison interface
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    
    # Get the list of cryptocurrencies
    crypto_options = list(data_fetcher.MAJOR_CRYPTOCURRENCIES.keys())
    
    # Select cryptocurrencies to compare
    st.markdown("<h3>Select Cryptocurrencies to Compare</h3>", unsafe_allow_html=True)
    selected_cryptos = st.multiselect(
        "Choose cryptocurrencies for comparison",
        options=crypto_options,
        default=["Bitcoin", "Ethereum"] if len(crypto_options) >= 2 else crypto_options[:1]
    )
    
    # Period selection
    period_options = {
        "1d": "1 Day",
        "5d": "5 Days",
        "1mo": "1 Month",
        "3mo": "3 Months",
        "6mo": "6 Months",
        "1y": "1 Year",
        "ytd": "Year to Date",
        "max": "Max"
    }
    
    selected_period = st.select_slider(
        "Select Time Period",
        options=list(period_options.keys()),
        value="1mo",
        format_func=lambda x: period_options[x]
    )
    
    if selected_cryptos:
        # Get data for selected cryptocurrencies
        crypto_data = {}
        crypto_symbols = []
        
        for crypto_name in selected_cryptos:
            symbol = data_fetcher.MAJOR_CRYPTOCURRENCIES[crypto_name]
            crypto_symbols.append(symbol)
            
            # Fetch data
            data = data_fetcher.get_crypto_data(symbol.split('-')[0], period=selected_period)
            if data is not None:
                crypto_data[symbol] = data
        
        if crypto_data:
            # Price performance comparison
            st.markdown("<h3>Price Performance Comparison</h3>", unsafe_allow_html=True)
            
            # Create price performance chart
            fig = go.Figure()
            
            for symbol, data in crypto_data.items():
                # Normalize to 100 at the start
                if not data.empty:
                    normalized_data = (data['Close'] / data['Close'].iloc[0]) * 100
                    
                    # Get the name from the symbol
                    crypto_name = next((name for name, sym in data_fetcher.MAJOR_CRYPTOCURRENCIES.items() if sym == symbol), symbol)
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=normalized_data,
                        name=crypto_name,
                        connectgaps=True
                    ))
            
            fig.update_layout(
                title="Normalized Price Performance (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalized Price",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics table
            metrics_data = []
            
            for symbol, data in crypto_data.items():
                if not data.empty:
                    # Calculate metrics
                    current_price = data['Close'].iloc[-1]
                    start_price = data['Close'].iloc[0]
                    change_pct = ((current_price - start_price) / start_price) * 100
                    
                    # Calculate volatility
                    returns = data['Close'].pct_change().dropna()
                    volatility = returns.std() * 100
                    
                    # Calculate max drawdown
                    rolling_max = data['Close'].cummax()
                    drawdown = ((data['Close'] - rolling_max) / rolling_max) * 100
                    max_drawdown = drawdown.min()
                    
                    # Get trading volume
                    avg_volume = data['Volume'].mean() if 'Volume' in data else None
                    
                    # Get the name from the symbol
                    crypto_name = next((name for name, sym in data_fetcher.MAJOR_CRYPTOCURRENCIES.items() if sym == symbol), symbol)
                    
                    metrics_data.append({
                        'Name': crypto_name,
                        'Symbol': symbol.split('-')[0],
                        'Current Price': current_price,
                        'Period Change %': change_pct,
                        'Volatility': volatility,
                        'Max Drawdown': max_drawdown,
                        'Avg Volume': avg_volume
                    })
            
            # Convert to DataFrame
            if metrics_data:
                df = pd.DataFrame(metrics_data)
                
                # Function to highlight performance
                def highlight_performance(val):
                    if val > 0:
                        return f'<span style="color:#4CAF50">+{val:.2f}%</span>'
                    elif val < 0:
                        return f'<span style="color:#FF5252">{val:.2f}%</span>'
                    else:
                        return f'<span style="color:#FFFFFF">{val:.2f}%</span>'
                
                # Format the data
                df['Current Price'] = df['Current Price'].apply(lambda x: f"${x:.2f}")
                df['Period Change %'] = df['Period Change %'].apply(highlight_performance)
                df['Volatility'] = df['Volatility'].apply(lambda x: f"{x:.2f}%")
                df['Max Drawdown'] = df['Max Drawdown'].apply(lambda x: f"{x:.2f}%")
                df['Avg Volume'] = df['Avg Volume'].apply(lambda x: utils.format_large_number(x) if pd.notnull(x) else "N/A")
                
                # Display the table
                st.markdown("<h3>Performance Metrics</h3>", unsafe_allow_html=True)
                st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Volatility comparison
                st.markdown("<h3>Volatility Comparison</h3>", unsafe_allow_html=True)
                
                # Create volatility bar chart
                volatility_data = [{'Name': row['Name'], 'Volatility': row['Volatility']} for row in metrics_data]
                volatility_df = pd.DataFrame(volatility_data)
                
                # Handle different types of volatility values
                # Convert all volatility values to float
                volatility_df['Volatility'] = volatility_df['Volatility'].apply(
                    lambda x: float(x.rstrip('%')) if isinstance(x, str) else float(x)
                )
                
                fig = px.bar(
                    volatility_df,
                    x='Name',
                    y='Volatility',
                    title="Volatility (Standard Deviation of Returns)",
                    color='Volatility',
                    color_continuous_scale=['green', 'yellow', 'red']
                )
                
                fig.update_layout(
                    height=400,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.1)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                st.markdown("<h3>Price Correlation</h3>", unsafe_allow_html=True)
                
                # Create a DataFrame with all closing prices
                close_data = {}
                
                for symbol, data in crypto_data.items():
                    if not data.empty:
                        crypto_name = next((name for name, sym in data_fetcher.MAJOR_CRYPTOCURRENCIES.items() if sym == symbol), symbol)
                        close_data[crypto_name] = data['Close']
                
                if close_data:
                    # Create a DataFrame from the collected data
                    close_df = pd.DataFrame(close_data)
                    
                    # Calculate correlation matrix
                    corr_matrix = close_df.corr()
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        title="Price Correlation Matrix",
                        color_continuous_scale='RdBu_r',
                        zmin=-1,
                        zmax=1
                    )
                    
                    fig.update_layout(
                        height=500,
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.1)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to fetch data for the selected cryptocurrencies. Please try different selections.")
    else:
        st.info("Please select at least one cryptocurrency to see the comparison.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def crypto_prediction():
    """Show cryptocurrency price prediction tool"""
    st.markdown("""
    <div class="dashboard-header">
        <h1>Cryptocurrency Price Prediction</h1>
        <p>Machine learning-based price forecasts for cryptocurrencies</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    
    # Import prediction module here to avoid circular imports
    import prediction
    
    # Select cryptocurrency and prediction parameters
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Cryptocurrency selection
        crypto_options = list(data_fetcher.MAJOR_CRYPTOCURRENCIES.keys())
        selected_crypto = st.selectbox(
            "Select Cryptocurrency",
            options=crypto_options,
            index=0
        )
        
        # Get symbol for the selected cryptocurrency
        selected_symbol = data_fetcher.MAJOR_CRYPTOCURRENCIES[selected_crypto]
        
        # Model selection
        model_type = st.selectbox(
            "Select Prediction Model",
            ["Linear Regression", "Random Forest", "Neural Network"],
            index=1
        )
    
    with col2:
        # Prediction days
        prediction_days = st.slider(
            "Prediction Horizon (Days)",
            min_value=1,
            max_value=60,
            value=30,
            step=1
        )
        
        # Historical data range
        data_range = st.select_slider(
            "Historical Data Range",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            value="1y"
        )
    
    # Get the historical data
    symbol_without_suffix = selected_symbol.split('-')[0]
    crypto_data = data_fetcher.get_crypto_data(symbol_without_suffix, period=data_range)
    
    if crypto_data is not None and not crypto_data.empty:
        # Display historical price chart
        st.markdown("<h3>Historical Price Data</h3>", unsafe_allow_html=True)
        
        fig = utils.create_candlestick_chart(crypto_data, title=f"{selected_crypto} ({symbol_without_suffix}) Historical Price")
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate predictions using the selected model
        if st.button("Generate Price Prediction"):
            with st.spinner(f"Generating {prediction_days}-day prediction using {model_type} model..."):
                # Call the prediction function from the prediction module
                prediction_result = prediction.predict_stock_prices(
                    crypto_data,
                    model_type.lower().replace(" ", "_"),
                    prediction_days
                )
                
                if prediction_result:
                    # The prediction function returns: predictions, dates, accuracy
                    predicted_data, prediction_dates, accuracy = prediction_result
                    
                    # Get the last known price
                    current_price = crypto_data['Close'].iloc[-1]
                    future_price = predicted_data[-1]
                    price_change = ((future_price - current_price) / current_price) * 100
                    
                    # Determine recommendation based on price change
                    if price_change > 15:
                        recommendation = "Strong Buy"
                    elif price_change > 5:
                        recommendation = "Buy"
                    elif price_change > -5:
                        recommendation = "Hold"
                    elif price_change > -15:
                        recommendation = "Sell"
                    else:
                        recommendation = "Strong Sell"
                    
                    # Display prediction results
                    st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
                    
                    # Price prediction metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Current Price",
                            f"${current_price:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            f"Predicted Price ({prediction_days} days)",
                            f"${future_price:.2f}",
                            f"{price_change:.2f}%"
                        )
                    
                    with col3:
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <h4>Recommendation</h4>
                            <div class="prediction-recommendation {recommendation.lower().replace(' ', '-')}">
                                {recommendation}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Create prediction visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=crypto_data.index,
                        y=crypto_data['Close'],
                        name="Historical",
                        line=dict(color='#4CAF50')
                    ))
                    
                    # Create future dates for predictions
                    last_date = crypto_data.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
                    
                    # Predicted data
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predicted_data,
                        name="Predicted",
                        line=dict(color='#1E88E5', dash='dash')
                    ))
                    
                    # Add confidence interval (for illustration)
                    crypto_close_values = crypto_data['Close'].values
                    std_dev = np.std(crypto_close_values[-30:]) if len(crypto_close_values) >= 30 else np.std(crypto_close_values)
                    upper_bound = predicted_data + 1.96 * std_dev
                    lower_bound = predicted_data - 1.96 * std_dev
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line_color='rgba(30, 136, 229, 0.1)',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(30, 136, 229, 0.1)',
                        line=dict(width=0),
                        name="95% Confidence Interval"
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{prediction_days}-Day Price Prediction for {selected_crypto}",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        template="plotly_dark",
                        height=500,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.1)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model performance metrics
                    st.markdown("<h3>Model Performance Metrics</h3>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Use accuracy value returned by the predict function
                        st.metric("Model Accuracy", f"{accuracy:.4f}")
                    
                    with col2:
                        # Display the model type used
                        st.metric("Model Type", model_type.upper())
                    
                    # Create prediction data with is_crypto flag
                    prediction_data = {
                        'symbol': selected_symbol,
                        'company_name': selected_crypto,
                        'current_price': current_price,
                        'predicted_price': future_price,
                        'days': prediction_days,
                        'model': model_type,
                        'recommendation': recommendation,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'is_crypto': True
                    }
                    
                    # Save prediction to history
                    database.save_prediction(st.session_state.username, prediction_data)
                    
                    st.success(f"Prediction completed and saved to your history.")
                else:
                    st.error("Unable to generate prediction. Please try a different cryptocurrency or model.")
    else:
        st.error(f"Unable to fetch historical data for {selected_crypto}. Please try a different cryptocurrency.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display prediction history
    st.markdown('<div class="content-container glass-container">', unsafe_allow_html=True)
    st.markdown("<h3>Your Prediction History</h3>", unsafe_allow_html=True)
    
    # Get prediction history for cryptocurrencies
    crypto_predictions = database.get_prediction_history(st.session_state.username, crypto_only=True)
    
    if crypto_predictions and len(crypto_predictions) > 0:
        # Convert to DataFrame for display
        df = pd.DataFrame(crypto_predictions)
        
        # Format columns
        if 'timestamp' in df.columns:
            df['Date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')
        if 'symbol' in df.columns:
            df['Symbol'] = df['symbol'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else x)
        if 'company_name' in df.columns:
            df['Name'] = df['company_name']
        if 'current_price' in df.columns:
            df['Current Price'] = df['current_price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
        if 'predicted_price' in df.columns:
            df['Predicted Price'] = df['predicted_price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
        if 'days' in df.columns:
            df['Period'] = df['days'].apply(lambda x: f"{x} days" if pd.notnull(x) else "N/A")
        if 'model' in df.columns:
            df['Model'] = df['model']
            
        # Calculate returns if both prices are available
        if 'current_price' in df.columns and 'predicted_price' in df.columns:
            df['Expected Return'] = ((df['predicted_price'] - df['current_price']) / df['current_price'] * 100)
            
            # Format returns with colorization
            def highlight_returns(val):
                if pd.isnull(val):
                    return "N/A"
                elif val > 0:
                    return f'<span style="color:#4CAF50">+{val:.2f}%</span>'
                elif val < 0:
                    return f'<span style="color:#FF5252">{val:.2f}%</span>'
                else:
                    return f'<span style="color:#FFFFFF">{val:.2f}%</span>'
                
            df['Expected Return'] = df['Expected Return'].apply(highlight_returns)
            
        if 'recommendation' in df.columns:
            # Format recommendation with colorization
            def color_recommendation(val):
                if pd.isnull(val):
                    return "N/A"
                elif val == "Strong Buy":
                    return f'<span style="color:#00C853">{val}</span>'
                elif val == "Buy":
                    return f'<span style="color:#4CAF50">{val}</span>'
                elif val == "Hold":
                    return f'<span style="color:#FFD600">{val}</span>'
                elif val == "Sell":
                    return f'<span style="color:#FF6D00">{val}</span>'
                elif val == "Strong Sell":
                    return f'<span style="color:#DD2C00">{val}</span>'
                else:
                    return val
                
            df['Recommendation'] = df['recommendation'].apply(color_recommendation)
        
        # Select and reorder columns for display
        display_columns = [col for col in ['Date', 'Symbol', 'Name', 'Current Price', 
                                        'Predicted Price', 'Period', 'Expected Return', 
                                        'Model', 'Recommendation'] if col in df.columns]
        
        # Show table
        if display_columns:
            display_df = df[display_columns].sort_values('Date', ascending=False)
            st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No prediction history data available.")
    else:
        st.info("You haven't made any cryptocurrency predictions yet. Try creating one with the form above.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def execute_crypto_order(username, symbol, name, quantity, price, order_type):
    """Execute a cryptocurrency buy or sell order"""
    # Get user data
    user_data = database.get_user_data(username)
    
    if user_data is None:
        st.error("User data not found.")
        return False
    
    # Initialize crypto portfolio if it doesn't exist
    if 'crypto_portfolio' not in user_data:
        user_data['crypto_portfolio'] = {}
    
    # Initialize crypto trading history if it doesn't exist
    if 'crypto_trading_history' not in user_data:
        user_data['crypto_trading_history'] = []
    
    # Calculate total amount
    total_amount = quantity * price
    timestamp = datetime.now().isoformat()
    
    # Handle the order
    if order_type == "Buy":
        # Check if user has enough balance
        # Ensure demo_crypto_balance exists in user data
        if 'demo_crypto_balance' not in user_data:
            user_data['demo_crypto_balance'] = 1000000.0
            
        demo_balance = user_data.get('demo_crypto_balance', 1000000.0)
        
        if total_amount > demo_balance:
            st.error("Insufficient crypto funds.")
            return False
        
        # Update user's balance
        user_data['demo_crypto_balance'] -= total_amount
        
        # Update portfolio
        if symbol not in user_data['crypto_portfolio']:
            user_data['crypto_portfolio'][symbol] = {
                'quantity': quantity,
                'avg_price': price
            }
        else:
            # Calculate new average cost
            current_quantity = user_data['crypto_portfolio'][symbol]['quantity']
            current_avg_price = user_data['crypto_portfolio'][symbol]['avg_price']
            
            total_quantity = current_quantity + quantity
            total_cost = (current_quantity * current_avg_price) + (quantity * price)
            
            user_data['crypto_portfolio'][symbol]['quantity'] = total_quantity
            user_data['crypto_portfolio'][symbol]['avg_price'] = total_cost / total_quantity
    
    elif order_type == "Sell":
        # Check if user has enough crypto
        if symbol not in user_data['crypto_portfolio'] or user_data['crypto_portfolio'][symbol]['quantity'] < quantity:
            st.error(f"Insufficient {symbol} balance.")
            return False
        
        # Ensure demo_crypto_balance exists in user data
        if 'demo_crypto_balance' not in user_data:
            user_data['demo_crypto_balance'] = 1000000.0
            
        # Update user's balance
        user_data['demo_crypto_balance'] += total_amount
        
        # Update portfolio
        user_data['crypto_portfolio'][symbol]['quantity'] -= quantity
        
        # Remove the crypto from portfolio if quantity is zero
        if user_data['crypto_portfolio'][symbol]['quantity'] <= 0:
            del user_data['crypto_portfolio'][symbol]
    
    # Record the trade in history
    user_data['crypto_trading_history'].append({
        'timestamp': timestamp,
        'symbol': symbol,
        'name': name,
        'quantity': quantity,
        'price': price,
        'total_amount': total_amount,
        'order_type': order_type,
        'auto_trade': False
    })
    
    # Save updated user data
    database.update_user_data(username, user_data)
    
    # Update session state with new balance value
    st.session_state.crypto_balance = user_data['demo_crypto_balance']
    
    return True