import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import data_fetcher
import utils

def show_comparison_page():
    """Display the stock comparison tool page"""
    if not st.session_state.logged_in:
        st.warning("Please login to access the stock comparison tool.")
        return
    
    st.header("Stock Comparison Tool")
    
    # Let user select multiple stocks to compare
    st.subheader("Select Stocks to Compare")
    
    # Allow user to enter multiple stock symbols
    default_symbols = []
    all_symbols_input = st.text_input(
        "Enter stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)", 
        value=""
    )
    
    # Parse input
    if all_symbols_input:
        symbols = [symbol.strip().upper() for symbol in all_symbols_input.split(',')]
    else:
        symbols = []
    
    # Popular stocks quick selection
    popular_stocks = {
        "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "Banking": ["JPM", "BAC", "WFC", "C", "GS"],
        "Healthcare": ["JNJ", "PFE", "MRK", "UNH", "ABT"],
        "Retail": ["WMT", "TGT", "COST", "HD", "AMZN"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"]
    }
    
    # Allow user to select from popular categories
    selected_category = st.selectbox("Or select from popular categories:", 
                                  ["Choose a category..."] + list(popular_stocks.keys()))
    
    if selected_category != "Choose a category...":
        preset_symbols = popular_stocks[selected_category]
        # Create buttons for quick addition
        cols = st.columns(len(preset_symbols))
        for i, (col, symbol) in enumerate(zip(cols, preset_symbols)):
            with col:
                if st.button(symbol, key=f"add_{symbol}"):
                    if symbol not in symbols:
                        symbols.append(symbol)
    
    if not symbols:
        st.info("Please enter at least one stock symbol to begin comparison.")
        return
    
    # Analysis parameters
    st.subheader("Comparison Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time period selection
        period_options = {
            "1 Week": 7, 
            "1 Month": 30, 
            "3 Months": 90, 
            "6 Months": 180, 
            "1 Year": 365,
            "3 Years": 1095,
            "5 Years": 1825
        }
        selected_period = st.selectbox("Select time period:", list(period_options.keys()))
        days = period_options[selected_period]
    
    with col2:
        # Analysis metrics selection
        metrics = st.multiselect(
            "Select metrics to compare:", 
            ["Price Performance", "Volume", "Volatility", "Moving Averages", "Relative Strength"],
            default=["Price Performance"]
        )
    
    # Start analysis
    if st.button("Compare Stocks"):
        with st.spinner(f"Analyzing {len(symbols)} stocks..."):
            # Fetch data for all symbols
            stocks_data = {}
            valid_symbols = []
            start_date = pd.Timestamp(datetime.now() - timedelta(days=days)).tz_localize(None)
            
            for symbol in symbols:
                try:
                    data = data_fetcher.get_stock_data(symbol, period=f"{days}d")
                    if data is not None and not data.empty:
                        # Handle timezone-aware index properly
                        if data.index.tz is not None:
                            compare_index = data.index.tz_localize(None)
                            filtered_data = data[compare_index >= start_date]
                        else:
                            filtered_data = data[data.index >= start_date]
                        
                        if not filtered_data.empty:
                            stocks_data[symbol] = filtered_data
                            valid_symbols.append(symbol)
                except Exception as e:
                    st.error(f"Error fetching data for {symbol}: {str(e)}")
            
            if not stocks_data:
                st.error("Could not retrieve data for any of the selected stocks.")
                return
            
            # Show analysis based on selected metrics
            if "Price Performance" in metrics:
                show_price_performance(stocks_data, valid_symbols, selected_period)
            
            if "Volume" in metrics:
                show_volume_comparison(stocks_data, valid_symbols)
            
            if "Volatility" in metrics:
                show_volatility_comparison(stocks_data, valid_symbols)
            
            if "Moving Averages" in metrics:
                show_moving_averages(stocks_data, valid_symbols)
            
            if "Relative Strength" in metrics:
                show_relative_strength(stocks_data, valid_symbols)
            
            # Show summary comparison table
            show_comparison_table(stocks_data, valid_symbols, selected_period)

def show_price_performance(stocks_data, symbols, period_name):
    """Show price performance comparison"""
    st.subheader("Price Performance Comparison")
    
    # Normalize prices to 100 at the beginning for comparison
    normalized_data = {}
    
    for symbol, data in stocks_data.items():
        if 'Close' in data.columns and not data.empty:
            start_price = data['Close'].iloc[0]
            normalized_data[symbol] = (data['Close'] / start_price) * 100
    
    # Create plot
    fig = go.Figure()
    
    for symbol in symbols:
        if symbol in normalized_data:
            fig.add_trace(go.Scatter(
                x=stocks_data[symbol].index,
                y=normalized_data[symbol],
                mode='lines',
                name=symbol,
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title='Normalized Price (%)',
        xaxis_title='Date',
        hovermode='x unified',
        title=f"Price Performance Over {period_name} (Normalized to 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate performance metrics
    performance_data = []
    
    for symbol in symbols:
        if symbol in normalized_data:
            data = stocks_data[symbol]
            perf = normalized_data[symbol].iloc[-1] - 100
            
            # Calculate additional metrics
            returns = data['Close'].pct_change().dropna()
            
            performance_data.append({
                'Symbol': symbol,
                'Performance (%)': perf,
                'Annualized Volatility (%)': returns.std() * np.sqrt(252) * 100,
                'Max Drawdown (%)': utils.calculate_drawdown(data['Close'])
            })
    
    if performance_data:
        # Convert to DataFrame and display
        perf_df = pd.DataFrame(performance_data)
        perf_df['Performance (%)'] = perf_df['Performance (%)'].round(2)
        perf_df['Annualized Volatility (%)'] = perf_df['Annualized Volatility (%)'].round(2)
        perf_df['Max Drawdown (%)'] = perf_df['Max Drawdown (%)'].round(2)
        
        # Sort by performance
        perf_df = perf_df.sort_values('Performance (%)', ascending=False)
        
        # Apply conditional formatting
        def highlight_performance(val):
            if isinstance(val, float) or (isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit()):
                val_float = float(val)
                if val_float > 0:
                    return 'color: green'
                elif val_float < 0:
                    return 'color: red'
            return ''
        
        st.dataframe(perf_df.style.applymap(highlight_performance, subset=['Performance (%)']), use_container_width=True)
        
        # Identify best performer
        if not perf_df.empty:
            best_performer = perf_df.iloc[0]['Symbol']
            best_perf = perf_df.iloc[0]['Performance (%)']
            
            if best_perf > 0:
                st.success(f"🏆 {best_performer} is the best performer with a {best_perf:.2f}% return.")
            else:
                st.warning(f"🏆 {best_performer} performed best but still had a {best_perf:.2f}% return.")

def show_volume_comparison(stocks_data, symbols):
    """Show volume comparison"""
    st.subheader("Trading Volume Comparison")
    
    # Create subplots for volume
    fig = make_subplots(rows=len(symbols), cols=1, 
                      subplot_titles=[f"{symbol} Trading Volume" for symbol in symbols],
                      shared_xaxes=True,
                      vertical_spacing=0.02)
    
    for i, symbol in enumerate(symbols):
        if symbol in stocks_data:
            data = stocks_data[symbol]
            
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name=symbol,
                marker_color='rgba(0, 0, 255, 0.3)'
            ), row=i+1, col=1)
    
    fig.update_layout(
        height=200 * len(symbols),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
        hovermode='x unified'
    )
    
    # Update y-axis titles
    for i in range(1, len(symbols) + 1):
        fig.update_yaxes(title_text="Volume", row=i, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate average daily volume
    volume_data = []
    
    for symbol in symbols:
        if symbol in stocks_data:
            data = stocks_data[symbol]
            avg_volume = data['Volume'].mean()
            max_volume = data['Volume'].max()
            min_volume = data['Volume'].min()
            
            # Calculate company_info
            company_info = data_fetcher.get_company_info(symbol)
            market_cap = company_info.get('marketCap', 0) / 1e9  # Convert to billions
            
            volume_data.append({
                'Symbol': symbol,
                'Avg Daily Volume': f"{int(avg_volume):,}",
                'Max Volume': f"{int(max_volume):,}",
                'Min Volume': f"{int(min_volume):,}",
                'Market Cap ($B)': f"{market_cap:.2f}"
            })
    
    if volume_data:
        vol_df = pd.DataFrame(volume_data)
        st.dataframe(vol_df, use_container_width=True)

def show_volatility_comparison(stocks_data, symbols):
    """Show volatility comparison"""
    st.subheader("Volatility Comparison")
    
    # Calculate daily returns and rolling volatility
    volatility_data = {}
    
    for symbol, data in stocks_data.items():
        if not data.empty:
            # Calculate daily returns
            data['Daily Return'] = data['Close'].pct_change() * 100
            # Calculate 21-day rolling volatility (annualized)
            data['Volatility (21d)'] = data['Daily Return'].rolling(window=21).std() * np.sqrt(252)
            volatility_data[symbol] = data
    
    # Create plot for rolling volatility
    fig = go.Figure()
    
    for symbol in symbols:
        if symbol in volatility_data:
            data = volatility_data[symbol]
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Volatility (21d)'],
                mode='lines',
                name=symbol,
                hovertemplate='%{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title='21-Day Rolling Volatility (%, Annualized)',
        xaxis_title='Date',
        hovermode='x unified',
        title="Historical Volatility Comparison",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate volatility metrics
    vol_metrics = []
    
    for symbol in symbols:
        if symbol in volatility_data:
            data = volatility_data[symbol]
            daily_returns = data['Daily Return'].dropna()
            
            vol_metrics.append({
                'Symbol': symbol,
                'Current Volatility (%)': data['Volatility (21d)'].iloc[-1] if not pd.isna(data['Volatility (21d)'].iloc[-1]) else 0,
                'Average Volatility (%)': daily_returns.std() * np.sqrt(252),
                'Max Daily Gain (%)': daily_returns.max(),
                'Max Daily Loss (%)': daily_returns.min(),
                'Positive Days (%)': (daily_returns > 0).mean() * 100
            })
    
    if vol_metrics:
        # Convert to DataFrame and display
        vol_df = pd.DataFrame(vol_metrics)
        # Round numeric columns
        numeric_cols = ['Current Volatility (%)', 'Average Volatility (%)', 
                        'Max Daily Gain (%)', 'Max Daily Loss (%)', 'Positive Days (%)']
        vol_df[numeric_cols] = vol_df[numeric_cols].round(2)
        
        # Sort by current volatility
        vol_df = vol_df.sort_values('Current Volatility (%)', ascending=True)
        
        st.dataframe(vol_df, use_container_width=True)
        
        # Show volatility interpretation
        st.info("""
        **Volatility Interpretation:**
        - Lower volatility (< 15%) typically indicates more stable stocks
        - Medium volatility (15-30%) is common for growth stocks
        - High volatility (> 30%) suggests higher risk stocks or market turbulence
        """)

def show_moving_averages(stocks_data, symbols):
    """Show moving averages comparison"""
    st.subheader("Moving Averages Analysis")
    
    # Select which moving averages to display
    ma_options = [20, 50, 200]
    selected_mas = st.multiselect(
        "Select moving averages to display:", 
        [f"{ma}-day MA" for ma in ma_options],
        default=["50-day MA", "200-day MA"]
    )
    
    selected_ma_values = [int(ma.split('-')[0]) for ma in selected_mas]
    
    # Create subplots for each stock
    fig = make_subplots(rows=len(symbols), cols=1, 
                      subplot_titles=[f"{symbol} Price and Moving Averages" for symbol in symbols],
                      shared_xaxes=True,
                      vertical_spacing=0.03)
    
    # Colors for moving averages
    ma_colors = {20: 'rgba(255, 165, 0, 0.8)', 50: 'rgba(255, 0, 0, 0.8)', 200: 'rgba(0, 128, 0, 0.8)'}
    
    for i, symbol in enumerate(symbols):
        if symbol in stocks_data:
            data = stocks_data[symbol].copy()
            
            # Add price
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name=f"{symbol} Price",
                line=dict(color='royalblue', width=1)
            ), row=i+1, col=1)
            
            # Add moving averages
            for ma in ma_options:
                if ma in selected_ma_values:
                    if len(data) >= ma:
                        data[f'MA{ma}'] = data['Close'].rolling(window=ma).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[f'MA{ma}'],
                            mode='lines',
                            name=f"{symbol} {ma}-day MA",
                            line=dict(color=ma_colors[ma], width=1.5)
                        ), row=i+1, col=1)
    
    fig.update_layout(
        height=300 * len(symbols),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    
    # Update y-axis titles
    for i in range(1, len(symbols) + 1):
        fig.update_yaxes(title_text="Price", row=i, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate moving average signals
    ma_signals = []
    
    for symbol in symbols:
        if symbol in stocks_data:
            data = stocks_data[symbol].copy()
            
            # Calculate moving averages
            if len(data) >= 200:
                data['MA20'] = data['Close'].rolling(window=20).mean()
                data['MA50'] = data['Close'].rolling(window=50).mean()
                data['MA200'] = data['Close'].rolling(window=200).mean()
                
                # Current price
                current_price = data['Close'].iloc[-1]
                
                # Current MA values
                ma20 = data['MA20'].iloc[-1]
                ma50 = data['MA50'].iloc[-1]
                ma200 = data['MA200'].iloc[-1]
                
                # Determine trend
                trend = "Uptrend" if current_price > ma200 else "Downtrend"
                
                # Generate signal
                if current_price > ma20 and current_price > ma50 and current_price > ma200:
                    signal = "Strong Buy"
                elif current_price > ma20 and current_price > ma50:
                    signal = "Buy"
                elif current_price < ma20 and current_price < ma50 and current_price < ma200:
                    signal = "Strong Sell"
                elif current_price < ma20 and current_price < ma50:
                    signal = "Sell"
                else:
                    signal = "Neutral"
                
                # Check for golden cross (50-day crosses above 200-day)
                prev_data = data.iloc[-30:-1]  # Check the last 30 days
                golden_cross = any((prev_data['MA50'] <= prev_data['MA200']) & 
                                  (prev_data['MA50'].shift(-1) > prev_data['MA200'].shift(-1)))
                
                # Check for death cross (50-day crosses below 200-day)
                death_cross = any((prev_data['MA50'] >= prev_data['MA200']) & 
                                 (prev_data['MA50'].shift(-1) < prev_data['MA200'].shift(-1)))
                
                # Add to signals
                ma_signals.append({
                    'Symbol': symbol,
                    'Current Price': f"${current_price:.2f}",
                    '20-day MA': f"${ma20:.2f}",
                    '50-day MA': f"${ma50:.2f}",
                    '200-day MA': f"${ma200:.2f}",
                    'Trend': trend,
                    'Signal': signal,
                    'Golden Cross (Last 30d)': "Yes" if golden_cross else "No",
                    'Death Cross (Last 30d)': "Yes" if death_cross else "No"
                })
    
    if ma_signals:
        ma_df = pd.DataFrame(ma_signals)
        
        # Function to color signals
        def color_signal(val):
            if val == "Strong Buy" or val == "Buy":
                return 'background-color: rgba(0, 255, 0, 0.2)'
            elif val == "Strong Sell" or val == "Sell":
                return 'background-color: rgba(255, 0, 0, 0.2)'
            else:
                return ''
        
        # Function to color trend
        def color_trend(val):
            if val == "Uptrend":
                return 'color: green'
            elif val == "Downtrend":
                return 'color: red'
            return ''
        
        # Function to color crosses
        def color_cross(val):
            if val == "Yes":
                return 'font-weight: bold'
            return ''
        
        st.dataframe(ma_df.style
                   .applymap(color_signal, subset=['Signal'])
                   .applymap(color_trend, subset=['Trend'])
                   .applymap(color_cross, subset=['Golden Cross (Last 30d)', 'Death Cross (Last 30d)']), 
                   use_container_width=True)
        
        # Moving average interpretation
        st.info("""
        **Moving Average Interpretation:**
        - **Golden Cross** (50-day crosses above 200-day) is considered a bullish signal
        - **Death Cross** (50-day crosses below 200-day) is considered a bearish signal
        - Price above longer-term moving averages generally indicates bullish momentum
        - Price below longer-term moving averages generally indicates bearish momentum
        """)

def show_relative_strength(stocks_data, symbols):
    """Show relative strength comparison against market (S&P 500)"""
    st.subheader("Relative Strength Analysis")
    
    # Get S&P 500 data for the same period
    try:
        sp500_data = data_fetcher.get_stock_data("^GSPC", period="1y")
        
        if sp500_data is None or sp500_data.empty:
            st.error("Could not fetch S&P 500 data for relative strength analysis.")
            return
        
        # Calculate relative strength
        relative_strength = {}
        
        for symbol in symbols:
            if symbol in stocks_data:
                stock_data = stocks_data[symbol].copy()
                
                # Align data
                aligned_dates = stock_data.index.intersection(sp500_data.index)
                if len(aligned_dates) < 10:  # Need at least 10 data points
                    continue
                
                stock_price = stock_data.loc[aligned_dates, 'Close']
                market_price = sp500_data.loc[aligned_dates, 'Close']
                
                # Normalize to starting value
                norm_stock = stock_price / stock_price.iloc[0]
                norm_market = market_price / market_price.iloc[0]
                
                # Calculate relative strength (stock/market)
                rs = norm_stock / norm_market
                relative_strength[symbol] = rs
        
        if not relative_strength:
            st.warning("Could not calculate relative strength for any of the selected stocks.")
            return
        
        # Create relative strength plot
        fig = go.Figure()
        
        # Add horizontal line at 1.0
        fig.add_shape(
            type="line",
            x0=min([rs.index[0] for rs in relative_strength.values()]),
            y0=1,
            x1=max([rs.index[-1] for rs in relative_strength.values()]),
            y1=1,
            line=dict(color="gray", width=1, dash="dash"),
        )
        
        for symbol, rs in relative_strength.items():
            fig.add_trace(go.Scatter(
                x=rs.index,
                y=rs,
                mode='lines',
                name=symbol,
                hovertemplate='%{y:.4f}<extra></extra>'
            ))
        
        fig.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            yaxis_title='Relative Strength vs S&P 500',
            xaxis_title='Date',
            hovermode='x unified',
            title="Relative Strength Comparison Against S&P 500",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate relative strength metrics
        rs_metrics = []
        
        for symbol, rs in relative_strength.items():
            # Calculate beta against market
            stock_data = stocks_data[symbol].copy()
            aligned_dates = stock_data.index.intersection(sp500_data.index)
            
            if len(aligned_dates) >= 20:
                stock_returns = stock_data.loc[aligned_dates, 'Close'].pct_change().dropna()
                market_returns = sp500_data.loc[aligned_dates, 'Close'].pct_change().dropna()
                
                # Align returns
                common_idx = stock_returns.index.intersection(market_returns.index)
                if len(common_idx) >= 20:
                    stock_returns = stock_returns.loc[common_idx]
                    market_returns = market_returns.loc[common_idx]
                    
                    # Calculate beta
                    beta = utils.calculate_beta(stock_returns, market_returns)
                    
                    # Calculate current relative strength
                    current_rs = rs.iloc[-1]
                    rs_change = ((rs.iloc[-1] / rs.iloc[0]) - 1) * 100
                    
                    # Classify strength
                    if current_rs > 1.1:
                        strength = "Strong Outperformer"
                    elif current_rs > 1:
                        strength = "Outperformer"
                    elif current_rs > 0.9:
                        strength = "Underperformer"
                    else:
                        strength = "Weak Underperformer"
                    
                    rs_metrics.append({
                        'Symbol': symbol,
                        'Current RS': round(current_rs, 2),
                        'RS Change (%)': round(rs_change, 2),
                        'Beta': round(beta, 2),
                        'Strength Classification': strength
                    })
        
        if rs_metrics:
            rs_df = pd.DataFrame(rs_metrics)
            
            # Sort by current RS
            rs_df = rs_df.sort_values('Current RS', ascending=False)
            
            # Function to color strength classification
            def color_strength(val):
                if val == "Strong Outperformer":
                    return 'background-color: rgba(0, 128, 0, 0.2)'
                elif val == "Outperformer":
                    return 'background-color: rgba(144, 238, 144, 0.2)'
                elif val == "Underperformer":
                    return 'background-color: rgba(255, 165, 0, 0.2)'
                else:
                    return 'background-color: rgba(255, 0, 0, 0.2)'
            
            # Function to color RS Change
            def color_rs_change(val):
                if isinstance(val, float) or (isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit()):
                    val_float = float(val)
                    if val_float > 0:
                        return 'color: green'
                    elif val_float < 0:
                        return 'color: red'
                return ''
            
            # Display with formatting
            st.dataframe(rs_df.style
                       .applymap(color_strength, subset=['Strength Classification'])
                       .applymap(color_rs_change, subset=['RS Change (%)']), 
                       use_container_width=True)
            
            # Relative strength interpretation
            st.info("""
            **Relative Strength Interpretation:**
            - Values > 1 indicate the stock is outperforming the S&P 500
            - Values < 1 indicate the stock is underperforming the S&P 500
            - Beta measures sensitivity to market movements:
              - Beta > 1: More volatile than the market
              - Beta < 1: Less volatile than the market
              - Beta near 0: Moves independently of the market
            """)
    
    except Exception as e:
        st.error(f"Error in relative strength analysis: {str(e)}")

def show_comparison_table(stocks_data, symbols, period_name):
    """Show a summary comparison table for all stocks"""
    st.subheader("Summary Comparison")
    
    # Fetch additional data for all symbols
    comparison_data = []
    
    for symbol in symbols:
        if symbol in stocks_data:
            try:
                data = stocks_data[symbol]
                
                # Get company info
                company_info = data_fetcher.get_company_info(symbol)
                
                # Calculate performance
                start_price = data['Close'].iloc[0]
                end_price = data['Close'].iloc[-1]
                perf = ((end_price / start_price) - 1) * 100
                
                # Calculate volatility
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                
                # Calculate max drawdown
                max_drawdown = utils.calculate_drawdown(data['Close'])
                
                # Get current stock data
                volume = data['Volume'].mean()
                
                # Compile data
                stock_summary = {
                    'Symbol': symbol,
                    'Name': company_info.get('shortName', symbol)[:20],
                    'Sector': company_info.get('sector', 'N/A'),
                    'Market Cap ($B)': company_info.get('marketCap', 0) / 1e9,
                    f'Return ({period_name})': perf,
                    'Volatility (%)': volatility,
                    'Max Drawdown (%)': max_drawdown,
                    'Avg Volume': volume,
                    'P/E Ratio': company_info.get('trailingPE', None),
                    'Dividend Yield (%)': company_info.get('dividendYield', 0) * 100 if company_info.get('dividendYield') else 0
                }
                
                comparison_data.append(stock_summary)
            except Exception as e:
                st.error(f"Error processing data for {symbol}: {str(e)}")
    
    if comparison_data:
        # Convert to DataFrame
        comp_df = pd.DataFrame(comparison_data)
        
        # Format numeric columns
        numeric_columns = ['Market Cap ($B)', f'Return ({period_name})', 
                          'Volatility (%)', 'Max Drawdown (%)', 'P/E Ratio', 'Dividend Yield (%)']
        
        for col in numeric_columns:
            if col in comp_df.columns:
                comp_df[col] = comp_df[col].round(2)
        
        # Format average volume
        if 'Avg Volume' in comp_df.columns:
            comp_df['Avg Volume'] = comp_df['Avg Volume'].apply(lambda x: f"{int(x):,}")
        
        # Sort by return
        comp_df = comp_df.sort_values(f'Return ({period_name})', ascending=False)
        
        # Function to color returns
        def color_returns(val):
            if isinstance(val, float) or (isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit()):
                val_float = float(val)
                if val_float > 0:
                    return 'color: green'
                elif val_float < 0:
                    return 'color: red'
            return ''
        
        # Display with formatting
        st.dataframe(comp_df.style.applymap(color_returns, subset=[f'Return ({period_name})']), 
                    use_container_width=True)