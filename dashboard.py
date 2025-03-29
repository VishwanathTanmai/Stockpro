import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import data_fetcher

def show_dashboard():
    """Display the main dashboard with market overview"""
    st.header("Market Dashboard")
    
    # Add market summary section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Major Market Indices")

        # Define market indices
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^FTSE": "FTSE 100",
            "^N225": "Nikkei 225"
        }

        # Fetch data for indices using yfinance
        def get_market_indices(symbols):
            data = []
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")  # Get daily data
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev_close = latest["Close"] - latest["Open"]
                    change_pct = (prev_close / latest["Open"]) * 100
                    data.append({"Symbol": symbol, "Price": latest["Close"], "Change %": change_pct})
                else:
                    data.append({"Symbol": symbol, "Price": "N/A", "Change %": "N/A"})
            return data

        # Get data
        indices_data = get_market_indices(list(indices.keys()))

        # Display indices data in a table
        if indices_data:
            indices_df = pd.DataFrame(indices_data)
            indices_df['Name'] = indices_df['Symbol'].map(indices)
            indices_df['Change %'] = indices_df['Change %'].map(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A")
            indices_df = indices_df[['Name', 'Price', 'Change %']]

            # Color code based on change percentage
            def highlight_change(val):
                if "%" not in val or val == "N/A":
                    return ""
                val_float = float(val.replace("%", ""))
                if val_float > 0:
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                elif val_float < 0:
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                else:
                    return ''

            st.dataframe(indices_df.style.map(highlight_change, subset=['Change %']))
        else:
            st.error("Failed to fetch market indices data")
    
    with col2:
        st.subheader("Market Overview")
        # Create a simple gauges/metrics for market sentiment
        market_data = data_fetcher.get_market_sentiment()
        
        if market_data:
            # Create three metrics for market overview
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Fear & Greed", market_data.get("fear_greed", "N/A"), 
                        market_data.get("fear_greed_change", "N/A"))
            col_b.metric("Trading Volume", f"{market_data.get('volume', 0)/1000000:.1f}M", 
                        f"{market_data.get('volume_change', 0):.1f}%")
            col_c.metric("Volatility", f"{market_data.get('volatility', 0):.2f}", 
                        f"{market_data.get('volatility_change', 0):.2f}%")
        else:
            st.error("Failed to load market overview data")
    
    # Add stock screener
    st.subheader("Quick Stock Search")
    stock_symbol = st.text_input("Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)", key="dashboard_stock_search")
    
    if stock_symbol:
        display_stock_info(stock_symbol)
    
    # Add trending stocks/most active
    st.subheader("Most Active Stocks Today")
    trending = data_fetcher.get_most_active_stocks()
    
    if trending is not None:
        # Limit to top 5 active stocks
        trending = trending[:5]
        
        # Display as cards in multiple columns
        cols = st.columns(len(trending))
        for i, (col, stock) in enumerate(zip(cols, trending)):
            with col:
                st.markdown(f"**{stock['Symbol']}**")
                st.write(f"{stock['Name'][:20]}...")
                price_color = "green" if stock['Change %'] > 0 else "red"
                st.markdown(f"${stock['Price']:.2f} <span style='color:{price_color}'>{stock['Change %']:.2f}%</span>", unsafe_allow_html=True)
    else:
        st.error("Failed to fetch trending stocks")
    
    # Add recent market news
    st.subheader("Latest Market News")
    news = data_fetcher.get_market_news()
    
    if news:
        for i, item in enumerate(news[:5]):  # Display latest 5 news items
            with st.expander(f"{item['title']}", expanded=False):
                st.write(f"**Source:** {item['source']}")
                st.write(f"**Published:** {item['published']}")
                st.write(item['summary'])
                st.markdown(f"[Read full article]({item['link']})")
    else:
        st.error("Failed to fetch market news")

def display_stock_info(symbol):
    """Display information about a specific stock"""
    symbol = symbol.upper().strip()
    stock_data = data_fetcher.get_stock_data(symbol)
    
    if stock_data is None:
        st.error(f"Could not find data for stock symbol: {symbol}")
        return
    
    # Basic info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        company_info = data_fetcher.get_company_info(symbol)
        st.subheader(company_info.get('longName', symbol))
        st.write(f"**Sector:** {company_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {company_info.get('industry', 'N/A')}")
        
        # Current price and change
        current_price = stock_data['Close'].iloc[-1]
        prev_price = stock_data['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        price_color = "green" if price_change >= 0 else "red"
        change_icon = "▲" if price_change >= 0 else "▼"
        
        st.markdown(f"""
        <div style='margin-top: 1rem;'>
            <span style='font-size: 1.8rem; font-weight: bold;'>${current_price:.2f}</span>
            <span style='font-size: 1.2rem; color: {price_color};'> {change_icon} {abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Key statistics
        st.write("**Key Statistics:**")
        stats = {
            "Market Cap": f"${company_info.get('marketCap', 0)/1e9:.2f}B",
            "P/E Ratio": f"{company_info.get('trailingPE', 'N/A')}",
            "52W High": f"${company_info.get('fiftyTwoWeekHigh', 0):.2f}",
            "52W Low": f"${company_info.get('fiftyTwoWeekLow', 0):.2f}",
            "Avg Volume": f"{company_info.get('averageVolume', 0)/1e6:.1f}M"
        }
        
        for key, value in stats.items():
            st.write(f"{key}: {value}")
    
    with col2:
        # Price chart
        st.subheader("Price History")
        
        # Date range selector for the chart
        time_options = {
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "5Y": 1825
        }
        
        selected_range = st.selectbox("Select time range:", list(time_options.keys()), index=1)
        days = time_options[selected_range]
        
        # Filter data based on selected range
        start_date = pd.Timestamp(datetime.now() - timedelta(days=days)).tz_localize(None)
        
        # Handle timezone-aware index properly
        if stock_data.index.tz is not None:
            compare_index = stock_data.index.tz_localize(None)
            filtered_data = stock_data[compare_index >= start_date]
        else:
            filtered_data = stock_data[stock_data.index >= start_date]
        
        # Initialize real_time_data for scope
        real_time_data = pd.DataFrame()
        
        # Add chart type selector
        chart_type = st.radio("Chart Type:", ["Line", "Candlestick", "Real-time"], horizontal=True)
        
        # Fetch real-time data if that option is selected
        if chart_type == "Real-time":
            ticker = yf.Ticker(symbol)
            interval = '1m'
            period = '1d'
            real_time_data = ticker.history(period=period, interval=interval)
        
        if chart_type == "Line":
            # Create price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='royalblue', width=2)
            ))
            
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis_title='Price (USD)',
                xaxis_title='Date',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif chart_type == "Candlestick":
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=filtered_data.index,
                open=filtered_data['Open'],
                high=filtered_data['High'],
                low=filtered_data['Low'],
                close=filtered_data['Close'],
                name='Price'
            )])
            
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis_title='Price (USD)',
                xaxis_title='Date',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Real-time chart
            # Create a container for the real-time chart
            real_time_container = st.empty()
            
            with st.spinner("Loading real-time data..."):
                # Get the most recent data with 1-minute intervals
                ticker = yf.Ticker(symbol)
                interval = '1m'
                period = '1d'
                real_time_data = ticker.history(period=period, interval=interval)
                
                if not real_time_data.empty:
                    # Create real-time chart
                    fig = go.Figure()
                    
                    # Price chart
                    fig.add_trace(go.Scatter(
                        x=real_time_data.index,
                        y=real_time_data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='royalblue', width=2)
                    ))
                    
                    # Add current time indicator
                    current_time = pd.Timestamp.now()
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis_title='Price (USD)',
                        xaxis_title='Time',
                        hovermode='x unified',
                        title=f"Real-time data for {symbol} (1-minute intervals)"
                    )
                    
                    # Display annotations for latest price
                    latest_price = real_time_data['Close'].iloc[-1]
                    fig.add_annotation(
                        x=real_time_data.index[-1],
                        y=latest_price,
                        text=f"${latest_price:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    )
                    
                    real_time_container.plotly_chart(fig, use_container_width=True)
                    
                    # Add auto-refresh button
                    if st.button("Refresh Data"):
                        st.rerun()
                    
                    # Display latest data statistics
                    st.markdown(f"""
                    **Latest Data:** {real_time_data.index[-1].strftime('%Y-%m-%d %H:%M')}  
                    **Latest Price:** ${latest_price:.2f}  
                    **Today's Range:** ${real_time_data['Low'].min():.2f} - ${real_time_data['High'].max():.2f}  
                    **Today's Volume:** {real_time_data['Volume'].sum():,}
                    """)
                else:
                    st.warning("Real-time data is not available for this stock. Try during market hours.")
                    # Fall back to regular chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='royalblue', width=2)
                    ))
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis_title='Price (USD)',
                        xaxis_title='Date',
                        hovermode='x unified'
                    )
                    
                    real_time_container.plotly_chart(fig, use_container_width=True)
        
        # Only show volume chart for Line and Candlestick views (not for real-time)
        if chart_type != "Real-time":
            # Trading volume as a small chart below price
            fig_volume = px.bar(
                filtered_data, 
                x=filtered_data.index, 
                y='Volume',
                color_discrete_sequence=['rgba(0, 0, 255, 0.3)']
            )
            
            fig_volume.update_layout(
                height=150,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis_title='Volume',
                xaxis_title=None,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        elif not real_time_data.empty:
            # Show real-time volume
            fig_volume = px.bar(
                real_time_data, 
                x=real_time_data.index, 
                y='Volume',
                color_discrete_sequence=['rgba(0, 0, 255, 0.3)']
            )
            
            fig_volume.update_layout(
                height=150,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis_title='Volume',
                xaxis_title=None,
                hovermode='x unified',
                title="Real-time Volume"
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
