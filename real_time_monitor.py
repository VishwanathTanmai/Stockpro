import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import yfinance as yf
from datetime import datetime, timedelta
import random

def show_real_time_monitor():
    """Display real-time stock price monitor with live charts"""
    st.markdown("<h1 style='text-align: center;'>Real-Time Market Monitor</h1>", unsafe_allow_html=True)
    
    # Settings container
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Symbol input with default popular stocks
            default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            symbol = st.selectbox(
                "Select Stock Symbol",
                options=default_symbols + ["Other"],
                index=0
            )
            
            if symbol == "Other":
                symbol = st.text_input("Enter Symbol", "").upper()
        
        with col2:
            # Time interval selection
            interval = st.selectbox(
                "Update Interval",
                options=["1 sec", "5 sec", "10 sec", "30 sec", "1 min"],
                index=1
            )
            
            # Convert to seconds
            interval_mapping = {"1 sec": 1, "5 sec": 5, "10 sec": 10, "30 sec": 30, "1 min": 60}
            interval_seconds = interval_mapping[interval]
        
        with col3:
            # Chart type
            chart_type = st.selectbox(
                "Display Type",
                options=["Line Chart", "Candlestick", "Area Chart"],
                index=0
            )
            
            # Display time window
            display_window = st.selectbox(
                "Display Window",
                options=["Last 5 min", "Last 15 min", "Last 30 min", "Last 1 hr"],
                index=1
            )
            
            # Convert to minutes
            window_mapping = {"Last 5 min": 5, "Last 15 min": 15, "Last 30 min": 30, "Last 1 hr": 60}
            window_minutes = window_mapping[display_window]
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize real-time data container
    if 'real_time_data' not in st.session_state:
        st.session_state.real_time_data = pd.DataFrame(columns=['Timestamp', 'Price', 'Change', 'Volume'])
        st.session_state.last_update = datetime.now()
        st.session_state.base_price = None
    
    # Main display
    chart_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Function to fetch real-time data
    def fetch_real_time_data(symbol):
        try:
            # Fetch current data from Yahoo Finance
            stock = yf.Ticker(symbol)
            current_data = stock.history(period="1d", interval="1m").iloc[-1]
            
            # Get current price and time
            current_price = current_data['Close']
            current_time = datetime.now()
            
            # If we don't have a base price, set it
            if st.session_state.base_price is None:
                st.session_state.base_price = current_price
            
            # Calculate change
            change = current_price - st.session_state.base_price
            change_pct = (change / st.session_state.base_price) * 100
            
            # For volume, use real volume or generate random for simulation
            volume = current_data['Volume'] / 1000  # Convert to K
            
            # Create new data entry
            new_data = pd.DataFrame({
                'Timestamp': [current_time],
                'Price': [current_price],
                'Change': [change],
                'Change_pct': [change_pct],
                'Volume': [volume]
            })
            
            # Append to existing data
            st.session_state.real_time_data = pd.concat([st.session_state.real_time_data, new_data], ignore_index=True)
            
            # Update last update time
            st.session_state.last_update = current_time
            
            # Trim data to only keep the display window
            window_start = current_time - timedelta(minutes=window_minutes)
            st.session_state.real_time_data = st.session_state.real_time_data[
                st.session_state.real_time_data['Timestamp'] > window_start
            ]
            
            return True, current_price, change_pct, volume
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False, None, None, None
    
    # Function to update chart
    def update_chart(data, symbol, chart_type):
        if data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        if chart_type == "Line Chart":
            fig.add_trace(go.Scatter(
                x=data['Timestamp'],
                y=data['Price'],
                mode='lines',
                name='Price',
                line=dict(color='#0099ff', width=2)
            ))
            
        elif chart_type == "Candlestick":
            # For candlestick, we need OHLC data
            # Since we only have real-time closing prices, we'll simulate the others
            # based on small random variations around the closing price
            
            # Group data by minute for candlesticks
            data['Minute'] = data['Timestamp'].dt.floor('min')
            
            candle_data = []
            for minute, group in data.groupby('Minute'):
                if len(group) > 0:
                    close = group['Price'].iloc[-1]
                    # Simulate open, high, low with small variations
                    spread = close * 0.0005  # 0.05% variation
                    open_price = group['Price'].iloc[0]
                    high = max(group['Price'].max(), close + spread)
                    low = min(group['Price'].min(), close - spread)
                    
                    candle_data.append({
                        'Timestamp': minute,
                        'Open': open_price,
                        'High': high,
                        'Low': low,
                        'Close': close
                    })
            
            if candle_data:
                candles_df = pd.DataFrame(candle_data)
                fig = go.Figure(data=[go.Candlestick(
                    x=candles_df['Timestamp'],
                    open=candles_df['Open'],
                    high=candles_df['High'],
                    low=candles_df['Low'],
                    close=candles_df['Close'],
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                )])
        
        elif chart_type == "Area Chart":
            fig.add_trace(go.Scatter(
                x=data['Timestamp'],
                y=data['Price'],
                mode='lines',
                fill='tozeroy',
                name='Price',
                line=dict(color='#0099ff', width=2),
                fillcolor='rgba(0, 153, 255, 0.2)'
            ))
        
        # Add volume as bar chart on secondary y-axis
        fig.add_trace(go.Bar(
            x=data['Timestamp'],
            y=data['Volume'],
            name='Volume (K)',
            yaxis='y2',
            marker=dict(color='rgba(200, 200, 200, 0.3)')
        ))
        
        # Set title and layout
        color = "#26a69a" if data['Change_pct'].iloc[-1] >= 0 else "#ef5350"
        title_text = f"{symbol} - Real-Time Price Movement"
        
        fig.update_layout(
            title=title_text,
            yaxis_title='Price ($)',
            xaxis_title='Time',
            height=500,
            template="plotly_dark",
            hovermode="x unified",
            yaxis=dict(domain=[0.3, 1.0]),
            yaxis2=dict(domain=[0, 0.25], title="Volume (K)", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        # Add indicator for price change direction
        change = data['Change_pct'].iloc[-1]
        direction = "▲" if change >= 0 else "▼"
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            text=f"<b>{direction} {abs(change):.2f}%</b>",
            font=dict(color=color, size=20),
            showarrow=False
        )
        
        return fig
    
    # Function to update stats display
    def update_stats(symbol, current_price, change_pct, volume):
        # Create a stats display with current price, change, and volume
        with stats_placeholder.container():
            st.markdown('<div class="css-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                change_color = "green" if change_pct >= 0 else "red"
                change_icon = "▲" if change_pct >= 0 else "▼"
                st.markdown(f"""
                <div style='text-align: center;'>
                    <p style='color: gray; margin-bottom: 5px;'>Change</p>
                    <p style='color: {change_color}; font-size: 1.5rem; font-weight: bold; margin-top: 0;'>
                        {change_icon} {abs(change_pct):.2f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric("Volume (K)", f"{volume:.2f}")
            
            with col4:
                # Add time since last update
                seconds_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
                st.markdown(f"""
                <div style='text-align: center;'>
                    <p style='color: gray; margin-bottom: 5px;'>Last Update</p>
                    <p style='font-size: 1.2rem; margin-top: 0;'>
                        {int(seconds_since_update)}s ago
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main execution loop
    if symbol and symbol != "Other":
        # Add a stop button
        stop_button = st.button("Stop Real-Time Updates")
        
        if not stop_button:
            # Fetch initial data
            success, current_price, change_pct, volume = fetch_real_time_data(symbol)
            
            if success:
                # Update chart
                chart = update_chart(st.session_state.real_time_data, symbol, chart_type)
                chart_placeholder.plotly_chart(chart, use_container_width=True)
                
                # Update stats
                update_stats(symbol, current_price, change_pct, volume)
                
                # Set an autorefresh time based on the interval
                st.write(f"Auto-updating every {interval} - Last update: {datetime.now().strftime('%H:%M:%S')}")
                st.empty()
                
                # Auto-refresh
                time.sleep(interval_seconds)
                st.rerun()
            else:
                st.error(f"Could not fetch data for {symbol}. Please check the symbol and try again.")
    else:
        st.info("Enter a valid stock symbol to start real-time monitoring.")
    
def create_trend_analysis():
    """Display trend analysis for selected stocks"""
    st.subheader("Market Trend Analysis")
    
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1,2])
        
        with col1:
            # Symbols selection
            default_indices = ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"]
            symbols = st.multiselect(
                "Select Indices & Stocks",
                options=default_indices + ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "Other"],
                default=default_indices[:3]
            )
            
            # Add custom symbol
            if "Other" in symbols:
                symbols.remove("Other")
                custom_symbol = st.text_input("Enter Custom Symbol", "").upper()
                if custom_symbol:
                    symbols.append(custom_symbol)
            
            # Period selection
            period = st.selectbox(
                "Time Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd"],
                index=2
            )
            
            # Analysis type
            analysis_type = st.selectbox(
                "Analysis Type",
                options=["Price Movement", "Relative Performance", "Volatility", "Correlation"],
                index=0
            )
            
            # Calculate button
            calculate = st.button("Analyze Trends", type="primary")
        
        with col2:
            if calculate and symbols:
                with st.spinner("Analyzing market trends..."):
                    try:
                        # Fetch data for all symbols
                        data = {}
                        for symbol in symbols:
                            stock = yf.Ticker(symbol)
                            data[symbol] = stock.history(period=period)
                        
                        # Create dataframe with close prices
                        close_df = pd.DataFrame({symbol: data[symbol]['Close'] for symbol in symbols})
                        
                        # Normalize for relative performance (divide by first value)
                        norm_df = close_df.div(close_df.iloc[0]) * 100
                        
                        # Analysis based on selected type
                        if analysis_type == "Price Movement":
                            fig = go.Figure()
                            for symbol in symbols:
                                fig.add_trace(go.Scatter(
                                    x=close_df.index,
                                    y=close_df[symbol],
                                    mode='lines',
                                    name=symbol
                                ))
                            fig.update_layout(
                                title="Price Movement",
                                yaxis_title="Price ($)",
                                template="plotly_dark",
                                height=400
                            )
                        
                        elif analysis_type == "Relative Performance":
                            fig = go.Figure()
                            for symbol in symbols:
                                fig.add_trace(go.Scatter(
                                    x=norm_df.index,
                                    y=norm_df[symbol],
                                    mode='lines',
                                    name=symbol
                                ))
                            fig.update_layout(
                                title="Relative Performance (Normalized)",
                                yaxis_title="Performance (%)",
                                template="plotly_dark",
                                height=400
                            )
                        
                        elif analysis_type == "Volatility":
                            # Calculate 10-day rolling volatility
                            vol_df = close_df.pct_change().rolling(10).std() * np.sqrt(252) * 100
                            
                            fig = go.Figure()
                            for symbol in symbols:
                                fig.add_trace(go.Scatter(
                                    x=vol_df.index,
                                    y=vol_df[symbol],
                                    mode='lines',
                                    name=symbol
                                ))
                            fig.update_layout(
                                title="Annualized Volatility (10-day Rolling)",
                                yaxis_title="Volatility (%)",
                                template="plotly_dark",
                                height=400
                            )
                        
                        elif analysis_type == "Correlation":
                            # Calculate correlation matrix
                            corr_matrix = close_df.pct_change().corr()
                            
                            # Create heatmap
                            fig = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.index,
                                colorscale='RdBu_r',
                                zmin=-1, zmax=1
                            ))
                            fig.update_layout(
                                title="Correlation Matrix",
                                template="plotly_dark",
                                height=400
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary statistics table
                        st.subheader("Summary Statistics")
                        returns = close_df.pct_change()
                        
                        stats = pd.DataFrame({
                            symbol: [
                                f"{data[symbol]['Close'].iloc[-1]:.2f}",
                                f"{(data[symbol]['Close'].iloc[-1] / data[symbol]['Close'].iloc[0] - 1) * 100:.2f}%",
                                f"{returns[symbol].std() * np.sqrt(252) * 100:.2f}%",
                                f"{returns[symbol].max() * 100:.2f}%",
                                f"{returns[symbol].min() * 100:.2f}%"
                            ] for symbol in symbols
                        }, index=["Current Price", "Period Return", "Annualized Volatility", "Max Daily Gain", "Max Daily Loss"])
                        
                        st.dataframe(stats, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error analyzing trends: {e}")
                        st.info("Please check that all symbols are valid and try again.")
            else:
                st.info("Select indices or stocks and click 'Analyze Trends' to begin trend analysis.")
        
        st.markdown('</div>', unsafe_allow_html=True)


def show_heatmap():
    """Display sector and market heatmap visualization"""
    st.subheader("Market Heatmap")
    
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Heatmap type selection
            heatmap_type = st.selectbox(
                "Heatmap Type",
                options=["S&P 500 Sectors", "NASDAQ 100", "Dow Jones 30", "Crypto Market"],
                index=0
            )
            
            # Metric selection
            metric = st.selectbox(
                "Performance Metric",
                options=["% Change", "Volume", "Market Cap"],
                index=0
            )
            
            # Time period
            period = st.selectbox(
                "Time Period",
                options=["1 Day", "5 Days", "1 Month", "3 Months", "YTD", "1 Year"],
                index=0
            )
            
            # Map period to yfinance format
            period_map = {
                "1 Day": "1d",
                "5 Days": "5d",
                "1 Month": "1mo",
                "3 Months": "3mo",
                "YTD": "ytd",
                "1 Year": "1y"
            }
            
            # Generate button
            generate = st.button("Generate Heatmap", type="primary")
        
        with col2:
            if generate:
                with st.spinner("Generating market heatmap..."):
                    try:
                        # Get appropriate symbols based on selection
                        if heatmap_type == "S&P 500 Sectors":
                            # Use sector ETFs to represent sectors
                            symbols = {
                                "Technology": "XLK",
                                "Healthcare": "XLV",
                                "Financials": "XLF",
                                "Energy": "XLE",
                                "Consumer Discretionary": "XLY",
                                "Consumer Staples": "XLP",
                                "Industrials": "XLI",
                                "Materials": "XLB",
                                "Utilities": "XLU",
                                "Real Estate": "XLRE",
                                "Communication Services": "XLC"
                            }
                        elif heatmap_type == "NASDAQ 100":
                            # Top NASDAQ components by weight
                            symbols = {
                                "Apple": "AAPL",
                                "Microsoft": "MSFT",
                                "Amazon": "AMZN",
                                "NVIDIA": "NVDA",
                                "Google (A)": "GOOGL",
                                "Meta": "META",
                                "Tesla": "TSLA",
                                "Google (C)": "GOOG",
                                "Adobe": "ADBE",
                                "Broadcom": "AVGO",
                                "Costco": "COST",
                                "PepsiCo": "PEP"
                            }
                        elif heatmap_type == "Dow Jones 30":
                            # Top Dow components
                            symbols = {
                                "Apple": "AAPL",
                                "Microsoft": "MSFT",
                                "Goldman Sachs": "GS",
                                "Home Depot": "HD",
                                "McDonald's": "MCD",
                                "Boeing": "BA",
                                "Caterpillar": "CAT",
                                "Visa": "V",
                                "JPMorgan": "JPM",
                                "Nike": "NKE",
                                "Disney": "DIS",
                                "Coca-Cola": "KO"
                            }
                        elif heatmap_type == "Crypto Market":
                            # Top cryptocurrencies
                            symbols = {
                                "Bitcoin": "BTC-USD",
                                "Ethereum": "ETH-USD",
                                "Binance Coin": "BNB-USD",
                                "Solana": "SOL-USD",
                                "XRP": "XRP-USD",
                                "Cardano": "ADA-USD",
                                "Avalanche": "AVAX-USD",
                                "Dogecoin": "DOGE-USD",
                                "Polkadot": "DOT-USD",
                                "Chainlink": "LINK-USD"
                            }
                        
                        # Fetch data
                        data = {}
                        for name, symbol in symbols.items():
                            ticker = yf.Ticker(symbol)
                            hist = ticker.history(period=period_map[period])
                            
                            if metric == "% Change":
                                # Calculate percentage change
                                value = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
                            elif metric == "Volume":
                                # Get average daily volume
                                value = hist['Volume'].mean() / 1000000  # In millions
                            elif metric == "Market Cap":
                                # Get market cap where available (might not work for all)
                                try:
                                    value = ticker.info.get('marketCap', 0) / 1000000000  # In billions
                                except:
                                    value = 0
                            
                            data[name] = value
                        
                        # Sort data for better visualization
                        sorted_data = sorted(data.items(), key=lambda x: x[1])
                        names = [item[0] for item in sorted_data]
                        values = [item[1] for item in sorted_data]
                        
                        # Create a colorscale based on values
                        # Green for positive, red for negative (for % change)
                        if metric == "% Change":
                            colors = ['#ef5350' if val < 0 else '#26a69a' for val in values]
                            colorscale = 'RdYlGn'
                        else:
                            colorscale = 'Viridis'
                            colors = None
                        
                        # Create label format based on metric
                        if metric == "% Change":
                            label_format = '.2f%'
                            title_suffix = f"% Change ({period})"
                        elif metric == "Volume":
                            label_format = '.2f'
                            title_suffix = f"Avg Daily Volume in M ({period})"
                        else:
                            label_format = '.2f'
                            title_suffix = f"Market Cap in B"
                        
                        # Create treemap figure
                        fig = go.Figure(go.Treemap(
                            labels=names,
                            parents=[""] * len(names),
                            values=values,
                            texttemplate="%{label}<br>%{value:" + label_format + "}",
                            marker=dict(
                                colors=colors if colors else values,
                                colorscale=colorscale,
                                showscale=True,
                                colorbar=dict(title=metric)
                            ),
                            hovertemplate='<b>%{label}</b><br>%{value:' + label_format + '}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title=f"{heatmap_type} - {title_suffix}",
                            template="plotly_dark",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add summary table
                        summary_df = pd.DataFrame({
                            'Name': names,
                            metric: values
                        })
                        
                        if metric == "% Change":
                            def color_pct_change(val):
                                color = '#26a69a' if val > 0 else '#ef5350'
                                return f'color: {color};'
                            
                            st.dataframe(summary_df.style.format({metric: '{:.2f}%'}).apply(lambda x: [color_pct_change(val) for val in x], axis=0, subset=[metric]), height=400)
                        else:
                            st.dataframe(summary_df.style.format({metric: '{:.2f}'}), height=400)
                    
                    except Exception as e:
                        st.error(f"Error generating heatmap: {e}")
                        st.info("Please try a different selection or time period.")
            else:
                st.info("Configure the heatmap options and click 'Generate Heatmap' to visualize market data.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_economic_indicators():
    """Display economic indicators dashboard"""
    st.subheader("Economic Indicators")
    
    with st.container():
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        
        # Info message about economic data
        st.info("This is a simulation of economic indicators. In a production app, this would connect to economic data APIs like FRED, World Bank, or trading platforms that provide economic data.")
        
        # Tabs for different economic categories
        tab1, tab2, tab3, tab4 = st.tabs(["Interest Rates", "GDP & Growth", "Inflation", "Employment"])
        
        with tab1:
            # Interest rates chart
            # Create exactly 63 months of data from Jan 2020 to March 2025
            date_range = pd.date_range(start='2020-01-01', periods=63, freq='MS')
            
            rates_data = {
                'Date': date_range,
                'Fed_Funds_Rate': [1.55, 1.58, 0.65, 0.05, 0.05, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09,
                                  0.08, 0.07, 0.07, 0.06, 0.06, 0.08, 0.10, 0.09, 0.08, 0.08, 0.08, 0.08,
                                  0.08, 0.08, 0.09, 0.33, 0.77, 1.21, 1.58, 2.18, 2.33, 3.08, 3.78, 4.33,
                                  4.57, 4.83, 5.08, 5.33, 5.33, 5.33, 5.33, 5.33, 5.33, 5.08, 4.83, 4.58,
                                  4.33, 4.08, 3.83, 3.58, 3.33, 3.08, 2.83, 2.58, 2.33, 2.08, 1.83, 1.58, 1.33],
                'Treasury_10Y': [1.76, 1.50, 0.70, 0.64, 0.65, 0.66, 0.55, 0.55, 0.68, 0.84, 0.84, 0.93,
                                1.08, 1.41, 1.74, 1.65, 1.58, 1.47, 1.30, 1.31, 1.52, 1.55, 1.43, 1.52,
                                1.78, 1.93, 2.32, 2.89, 2.93, 3.01, 2.90, 3.15, 3.80, 4.05, 3.88, 3.88,
                                4.10, 4.30, 4.57, 4.63, 4.60, 4.55, 4.50, 4.52, 4.40, 4.25, 4.10, 3.90,
                                3.75, 3.60, 3.45, 3.30, 3.15, 3.00, 2.90, 2.80, 2.70, 2.60, 2.50, 2.40, 2.35],
                'Mortgage_30Y': [3.72, 3.47, 3.30, 3.30, 3.16, 3.16, 3.02, 2.94, 2.89, 2.83, 2.77, 2.68,
                                2.74, 2.97, 3.08, 3.06, 2.96, 2.98, 2.87, 2.84, 2.90, 3.03, 3.07, 3.10,
                                3.45, 3.76, 4.17, 5.10, 5.23, 5.52, 5.22, 5.52, 6.11, 6.71, 7.08, 7.22,
                                7.31, 7.22, 7.10, 7.03, 6.90, 6.69, 6.61, 6.67, 6.49, 6.28, 6.18, 5.97,
                                5.89, 5.70, 5.60, 5.50, 5.40, 5.30, 5.20, 5.10, 5.00, 4.90, 4.80, 4.70, 4.65]
            }
            
            rates_df = pd.DataFrame(rates_data)
            rates_df.set_index('Date', inplace=True)
            
            # Create rates chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rates_df.index, y=rates_df['Fed_Funds_Rate'], mode='lines', name='Fed Funds Rate'))
            fig.add_trace(go.Scatter(x=rates_df.index, y=rates_df['Treasury_10Y'], mode='lines', name='10Y Treasury'))
            fig.add_trace(go.Scatter(x=rates_df.index, y=rates_df['Mortgage_30Y'], mode='lines', name='30Y Mortgage Rate'))
            
            fig.update_layout(
                title="U.S. Interest Rates (2020-2025)",
                yaxis_title="Rate (%)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Key rates table
            st.subheader("Current Key Rates")
            current_rates = {
                'Rate Type': ['Federal Funds Rate', '10-Year Treasury', '30-Year Mortgage', '2-Year Treasury', 'SOFR'],
                'Current Rate': [1.33, 2.35, 4.65, 2.11, 1.30],
                'Previous': [1.58, 2.40, 4.70, 2.23, 1.32],
                'Change': [-0.25, -0.05, -0.05, -0.12, -0.02]
            }
            
            rate_df = pd.DataFrame(current_rates)
            
            def color_change(val):
                color = '#26a69a' if val < 0 else '#ef5350' if val > 0 else 'gray'
                return f'color: {color}'
            
            st.dataframe(rate_df.style.format({'Current Rate': '{:.2f}%', 'Previous': '{:.2f}%', 'Change': '{:.2f}%'})
                         .apply(lambda x: [color_change(val) for val in x], axis=0, subset=['Change']), height=300)
            
        with tab2:
            # GDP and growth data
            # Create exactly 22 quarterly periods from Q1 2020 to Q2 2025
            date_range_gdp = pd.date_range(start='2020-01-01', periods=22, freq='QS')
            
            gdp_data = {
                'Date': date_range_gdp,
                'GDP_Growth': [-4.8, -31.2, 33.8, 4.5, 6.3, 6.7, 2.3, 6.9, -1.6, -0.6, 3.2, 2.6, 2.0, 1.4, 3.1, 3.3, 3.0, 2.8, 2.7, 2.6, 2.5, 2.4],
                'GDP_Value': [21.48, 19.52, 21.14, 21.48, 22.06, 22.74, 23.20, 24.00, 24.28, 24.85, 25.72, 26.14, 26.65, 27.06, 27.35, 27.68, 28.10, 28.56, 29.02, 29.48, 29.95, 30.42]
            }
            
            gdp_df = pd.DataFrame(gdp_data)
            gdp_df.set_index('Date', inplace=True)
            
            # Create GDP chart with dual axis
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=gdp_df.index,
                y=gdp_df['GDP_Growth'],
                name='GDP Growth (%)',
                marker_color=['#ef5350' if x < 0 else '#26a69a' for x in gdp_df['GDP_Growth']]
            ))
            
            fig.add_trace(go.Scatter(
                x=gdp_df.index,
                y=gdp_df['GDP_Value'],
                mode='lines',
                name='GDP Value (Trillion $)',
                yaxis='y2',
                line=dict(color='#ffab40', width=3)
            ))
            
            fig.update_layout(
                title="U.S. GDP Growth and Value (2020-2025)",
                yaxis=dict(
                    title="Quarterly Growth (%)",
                    range=[-35, 35]
                ),
                yaxis2=dict(
                    title="GDP Value (Trillion $)",
                    overlaying="y",
                    side="right",
                    range=[18, 32]
                ),
                template="plotly_dark",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Economic growth forecast
            st.subheader("Economic Growth Projections")
            
            forecast_data = {
                'Region': ['United States', 'Eurozone', 'China', 'Japan', 'India', 'UK', 'Brazil', 'Global'],
                '2024 Forecast': [2.4, 0.9, 4.5, 0.8, 6.2, 0.5, 2.0, 3.1],
                '2025 Forecast': [2.0, 1.2, 4.2, 1.0, 6.5, 1.3, 2.2, 3.2],
                'Change': [-0.4, 0.3, -0.3, 0.2, 0.3, 0.8, 0.2, 0.1]
            }
            
            forecast_df = pd.DataFrame(forecast_data)
            
            st.dataframe(forecast_df.style.format({'2024 Forecast': '{:.1f}%', '2025 Forecast': '{:.1f}%', 'Change': '{:.1f}%'})
                         .apply(lambda x: [color_change(val) for val in x], axis=0, subset=['Change']), height=300)
            
        with tab3:
            # Inflation data
            # Create exactly 63 months of data from Jan 2020 to March 2025
            date_range_inflation = pd.date_range(start='2020-01-01', periods=63, freq='MS')
            
            inflation_data = {
                'Date': date_range_inflation,
                'CPI': [2.3, 2.3, 1.5, 0.3, 0.1, 0.6, 1.0, 1.3, 1.4, 1.2, 1.2, 1.4,
                       1.4, 1.7, 2.6, 4.2, 5.0, 5.4, 5.4, 5.3, 5.4, 6.2, 6.8, 7.0,
                       7.5, 7.9, 8.5, 8.3, 8.6, 9.1, 8.5, 8.3, 8.2, 7.7, 7.1, 6.5,
                       6.4, 6.0, 5.0, 4.9, 4.0, 3.7, 3.7, 3.2, 3.2, 3.2, 3.1, 3.0,
                       2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.3, 2.2, 2.2, 2.1, 2.1, 2.1],
                'Core_CPI': [2.3, 2.4, 2.1, 1.4, 1.2, 1.2, 1.6, 1.7, 1.7, 1.6, 1.6, 1.6,
                            1.3, 1.3, 1.6, 3.0, 3.8, 4.5, 4.3, 4.0, 4.0, 4.6, 4.9, 5.5,
                            6.0, 6.4, 6.5, 6.2, 6.0, 5.9, 5.9, 6.3, 6.6, 6.3, 6.0, 5.7,
                            5.6, 5.5, 5.3, 5.1, 4.8, 4.3, 4.1, 4.1, 4.0, 3.8, 3.6, 3.5,
                            3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2],
                'PPI': [2.1, 1.2, -1.2, -1.8, -0.9, -0.1, 0.5, 0.5, 0.5, 0.5, 0.8, 0.8,
                       1.7, 2.8, 4.1, 6.2, 6.5, 6.6, 7.3, 8.3, 8.6, 8.8, 9.7, 9.8,
                       10.0, 11.2, 11.5, 11.0, 10.9, 11.3, 9.8, 8.7, 8.5, 8.0, 7.4, 6.5,
                       6.0, 4.9, 4.6, 3.3, 2.8, 2.2, 2.2, 1.9, 1.6, 1.5, 1.4, 1.3,
                       1.3, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9]
            }
            
            inflation_df = pd.DataFrame(inflation_data)
            inflation_df.set_index('Date', inplace=True)
            
            # Create inflation chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=inflation_df.index, y=inflation_df['CPI'], mode='lines', name='CPI (All Items)'))
            fig.add_trace(go.Scatter(x=inflation_df.index, y=inflation_df['Core_CPI'], mode='lines', name='Core CPI (ex. Food & Energy)'))
            fig.add_trace(go.Scatter(x=inflation_df.index, y=inflation_df['PPI'], mode='lines', name='Producer Price Index'))
            
            # Add Fed target line
            fig.add_shape(type="line",
                x0=min(inflation_df.index), y0=2.0, x1=max(inflation_df.index), y1=2.0,
                line=dict(color="yellow", width=2, dash="dash"),
            )
            
            fig.add_annotation(
                x=pd.Timestamp('2020-06-01'), y=2.2,
                text="Fed Target (2%)",
                showarrow=False,
                font=dict(color="yellow")
            )
            
            fig.update_layout(
                title="U.S. Inflation Rates (2020-2025)",
                yaxis_title="Rate (%)",
                template="plotly_dark",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Inflation components
            st.subheader("Current CPI Components (Year-over-Year % Change)")
            
            components_data = {
                'Component': ['Food', 'Housing', 'Transportation', 'Energy', 'Medical Care', 'Education', 'Apparel', 'Recreation'],
                'YoY Change': [2.8, 3.0, 1.5, 0.6, 2.2, 2.9, 0.5, 1.3],
                'Previous': [3.1, 3.2, 1.8, 0.9, 2.3, 3.0, 0.7, 1.5],
                'Weight (%)': [15.0, 42.7, 17.3, 7.0, 8.7, 3.0, 2.5, 3.8]
            }
            
            components_df = pd.DataFrame(components_data)
            
            def color_inflation(val):
                if val > 3.0:
                    return 'color: #ef5350; font-weight: bold'
                elif val > 2.5:
                    return 'color: #ff9800'
                elif val < 1.0:
                    return 'color: #26a69a'
                else:
                    return 'color: white'
            
            st.dataframe(components_df.style.format({'YoY Change': '{:.1f}%', 'Previous': '{:.1f}%', 'Weight (%)': '{:.1f}%'})
                         .map(color_inflation, subset=['YoY Change']), height=300)
            
        with tab4:
            # Employment data
            # Create exactly 63 months of data from Jan 2020 to March 2025
            date_range_employment = pd.date_range(start='2020-01-01', periods=63, freq='MS')
            
            employment_data = {
                'Date': date_range_employment,
                'Unemployment': [3.5, 3.5, 4.4, 14.7, 13.2, 11.0, 10.2, 8.4, 7.9, 6.9, 6.7, 6.7,
                                6.4, 6.2, 6.0, 6.0, 5.8, 5.9, 5.4, 5.2, 4.8, 4.6, 4.2, 3.9,
                                4.0, 3.8, 3.6, 3.6, 3.6, 3.6, 3.5, 3.7, 3.5, 3.7, 3.6, 3.5,
                                3.4, 3.4, 3.5, 3.8, 3.8, 3.8, 3.9, 3.9, 3.7, 3.7, 3.8, 3.9,
                                4.0, 4.0, 4.1, 4.1, 4.2, 4.2, 4.3, 4.3, 4.2, 4.2, 4.1, 4.1, 4.0],
                'Nonfarm_Payrolls': [-701, -2685, -20679, -1373, 4781, 1726, 1583, 1371, 649, 264, -306, 233,
                                    568, 536, 785, 614, 389, 962, 1091, 483, 379, 647, 588, 504,
                                    714, 398, 368, 386, 293, 293, 537, 315, 292, 263, 182, 254,
                                    472, 217, 169, 105, 179, 297, 163, 119, 262, 165, 216, 164,
                                    160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100],
                'Labor_Force_Participation': [63.3, 63.3, 62.6, 60.1, 60.6, 61.5, 61.5, 61.7, 61.5, 61.6, 61.5, 61.5,
                                            61.5, 61.5, 61.5, 61.6, 61.6, 61.6, 61.7, 61.8, 61.7, 61.9, 61.9, 61.9,
                                            62.2, 62.3, 62.2, 62.2, 62.3, 62.2, 62.1, 62.2, 62.1, 62.2, 62.2, 62.3,
                                            62.4, 62.5, 62.6, 62.5, 62.5, 62.5, 62.7, 62.8, 62.7, 62.7, 62.6, 62.6,
                                            62.5, 62.5, 62.4, 62.4, 62.3, 62.3, 62.2, 62.2, 62.1, 62.1, 62.0, 62.0, 62.0]
            }
            
            employment_df = pd.DataFrame(employment_data)
            employment_df.set_index('Date', inplace=True)
            
            # Create employment chart with dual axis
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=employment_df.index,
                y=employment_df['Unemployment'],
                mode='lines',
                name='Unemployment Rate (%)',
                line=dict(color='#ef5350', width=3)
            ))
            
            fig.add_trace(go.Bar(
                x=employment_df.index,
                y=employment_df['Nonfarm_Payrolls'],
                name='Nonfarm Payrolls (thousands)',
                marker_color=['#ef5350' if x < 0 else '#26a69a' for x in employment_df['Nonfarm_Payrolls']],
                yaxis='y2'
            ))
            
            fig.add_trace(go.Scatter(
                x=employment_df.index,
                y=employment_df['Labor_Force_Participation'],
                mode='lines',
                name='Labor Force Participation Rate (%)',
                line=dict(color='#ffab40', width=3)
            ))
            
            fig.update_layout(
                title="U.S. Employment Metrics (2020-2025)",
                yaxis=dict(
                    title="Rate (%)",
                    range=[2.5, 15.5]
                ),
                yaxis2=dict(
                    title="Nonfarm Payrolls (thousands)",
                    overlaying="y",
                    side="right",
                    range=[-21000, 5000]
                ),
                template="plotly_dark",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Employment by sector
            st.subheader("Employment by Sector")
            
            sector_data = {
                'Sector': ['Professional & Business', 'Education & Health', 'Retail Trade', 'Leisure & Hospitality', 
                         'Manufacturing', 'Construction', 'Financial Activities', 'Information Technology'],
                'Current (millions)': [22.5, 25.1, 15.8, 16.3, 12.9, 7.8, 8.9, 3.1],
                'YoY Change (%)': [1.5, 1.8, 0.2, 2.3, -0.3, -0.2, 0.5, 3.1],
                'Avg Hourly Wage ($)': [38.75, 33.20, 24.50, 21.80, 31.60, 35.10, 42.30, 52.80]
            }
            
            sector_df = pd.DataFrame(sector_data)
            
            def color_job_growth(val):
                color = '#26a69a' if val > 0 else '#ef5350'
                return f'color: {color};'
            
            st.dataframe(sector_df.style.format({'Current (millions)': '{:.1f}', 'YoY Change (%)': '{:.1f}%', 'Avg Hourly Wage ($)': '${:.2f}'})
                         .map(color_job_growth, subset=['YoY Change (%)']), height=300)
        
        st.markdown('</div>', unsafe_allow_html=True)