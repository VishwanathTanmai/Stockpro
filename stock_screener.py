import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import data_fetcher
import utils

def show_screener_page():
    """Display the stock screener tool page"""
    if not st.session_state.logged_in:
        st.warning("Please login to access the stock screener tool.")
        return
    
    st.header("Stock Screener")
    
    # Screener parameters
    st.subheader("Set Screening Criteria")
    
    # Choose screener type
    screener_type = st.radio(
        "Select screener mode:",
        ["Fundamental Screener", "Technical Screener", "Pre-built Strategies"],
        horizontal=True
    )
    
    if screener_type == "Fundamental Screener":
        show_fundamental_screener()
    elif screener_type == "Technical Screener":
        show_technical_screener()
    else:
        show_strategy_screener()

def show_fundamental_screener():
    """Display fundamental screener options"""
    # Sectors selection
    sectors = [
        "All Sectors", "Technology", "Healthcare", "Financials", 
        "Consumer Cyclical", "Consumer Defensive", "Industrials",
        "Basic Materials", "Energy", "Utilities", "Real Estate", "Communication Services"
    ]
    
    selected_sectors = st.multiselect(
        "Select sectors:", 
        sectors,
        default=["All Sectors"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market cap filter
        market_cap_options = [
            "Any Market Cap",
            "Mega Cap (> $200B)",
            "Large Cap ($10B - $200B)",
            "Mid Cap ($2B - $10B)",
            "Small Cap ($300M - $2B)",
            "Micro Cap (< $300M)"
        ]
        market_cap = st.selectbox("Market Cap", market_cap_options)
        
        # P/E Ratio filter
        pe_col1, pe_col2 = st.columns(2)
        with pe_col1:
            min_pe = st.number_input("Min P/E Ratio", value=0.0, step=1.0)
        with pe_col2:
            max_pe = st.number_input("Max P/E Ratio", value=100.0, step=1.0)
        
        # Dividend Yield filter
        div_col1, div_col2 = st.columns(2)
        with div_col1:
            min_dividend = st.number_input("Min Dividend (%)", value=0.0, step=0.1)
        with div_col2:
            max_dividend = st.number_input("Max Dividend (%)", value=20.0, step=0.1)
    
    with col2:
        # Price filter
        price_col1, price_col2 = st.columns(2)
        with price_col1:
            min_price = st.number_input("Min Price ($)", value=0.0, step=1.0)
        with price_col2:
            max_price = st.number_input("Max Price ($)", value=1000.0, step=10.0)
        
        # Revenue Growth filter
        rev_col1, rev_col2 = st.columns(2)
        with rev_col1:
            min_revenue_growth = st.number_input("Min Revenue Growth (%)", value=0.0, step=5.0)
        with rev_col2:
            include_profitable_only = st.checkbox("Profitable Companies Only", value=True)
        
        # Analyst Rating filter
        analyst_ratings = ["Any Rating", "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"]
        analyst_rating = st.selectbox("Minimum Analyst Rating", analyst_ratings)
    
    # Advanced options (hidden by default)
    with st.expander("Advanced Criteria"):
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            # Debt to Equity filter
            debt_col1, debt_col2 = st.columns(2)
            with debt_col1:
                min_debt_to_equity = st.number_input("Min Debt/Equity", value=0.0, step=0.1)
            with debt_col2:
                max_debt_to_equity = st.number_input("Max Debt/Equity", value=2.0, step=0.1)
            
            # Return on Equity filter
            min_roe = st.number_input("Min Return on Equity (%)", value=0.0, step=1.0)
        
        with adv_col2:
            # Profit Margin filter
            margin_col1, margin_col2 = st.columns(2)
            with margin_col1:
                min_profit_margin = st.number_input("Min Profit Margin (%)", value=0.0, step=1.0)
            with margin_col2:
                max_profit_margin = st.number_input("Max Profit Margin (%)", value=50.0, step=1.0)
            
            # PEG Ratio filter
            min_peg = st.number_input("Max PEG Ratio", value=0.0, step=0.1)
    
    # Stock universe selection
    universe_options = ["S&P 500", "Dow 30", "Nasdaq 100", "Russell 1000", "Custom List"]
    stock_universe = st.selectbox("Stock Universe", universe_options)
    
    if stock_universe == "Custom List":
        custom_symbols = st.text_input("Enter comma-separated stock symbols (e.g., AAPL, MSFT, GOOGL)")
        symbols_list = [symbol.strip().upper() for symbol in custom_symbols.split(',')] if custom_symbols else []
    else:
        symbols_list = get_index_components(stock_universe)
    
    # Run screener
    if st.button("Run Fundamental Screener"):
        if not symbols_list:
            st.warning("Please select a stock universe or enter custom stock symbols.")
            return
        
        # Run the screener
        with st.spinner(f"Screening {len(symbols_list)} stocks..."):
            results = run_fundamental_screener(
                symbols_list,
                selected_sectors,
                market_cap,
                (min_pe, max_pe),
                (min_dividend, max_dividend),
                (min_price, max_price),
                min_revenue_growth,
                include_profitable_only,
                analyst_rating,
                (min_debt_to_equity, max_debt_to_equity),
                min_roe,
                (min_profit_margin, max_profit_margin),
                min_peg
            )
            
            display_screener_results(results)

def show_technical_screener():
    """Display technical screener options"""
    # Time period selection
    period_options = {
        "1 Week": 7, 
        "1 Month": 30, 
        "3 Months": 90, 
        "6 Months": 180, 
        "1 Year": 365
    }
    period = st.selectbox("Lookback Period", list(period_options.keys()))
    days = period_options[period]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance filter
        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            min_performance = st.number_input("Min Performance (%)", value=-100.0, step=5.0)
        with perf_col2:
            max_performance = st.number_input("Max Performance (%)", value=100.0, step=5.0)
        
        # Volatility filter
        vol_col1, vol_col2 = st.columns(2)
        with vol_col1:
            min_volatility = st.number_input("Min Volatility (%)", value=0.0, step=5.0)
        with vol_col2:
            max_volatility = st.number_input("Max Volatility (%)", value=100.0, step=5.0)
        
        # Volume filter
        min_avg_volume = st.number_input("Min Avg Daily Volume", value=100000, step=100000, format="%d")
    
    with col2:
        # Moving average filter
        ma_options = [
            "Any MA Relationship",
            "Price above 50-day MA",
            "Price below 50-day MA",
            "Price above 200-day MA",
            "Price below 200-day MA",
            "50-day MA above 200-day MA",
            "50-day MA below 200-day MA",
            "Golden Cross (50-day crossing above 200-day)",
            "Death Cross (50-day crossing below 200-day)"
        ]
        ma_filter = st.selectbox("Moving Average Filter", ma_options)
        
        # RSI filter
        rsi_col1, rsi_col2 = st.columns(2)
        with rsi_col1:
            min_rsi = st.number_input("Min RSI", value=0, step=1, min_value=0, max_value=100)
        with rsi_col2:
            max_rsi = st.number_input("Max RSI", value=100, step=1, min_value=0, max_value=100)
        
        # Trend filter
        trend_options = ["Any Trend", "Uptrend", "Downtrend", "Sideways"]
        trend_filter = st.selectbox("Trend Filter", trend_options)
    
    # Advanced options (hidden by default)
    with st.expander("Advanced Technical Criteria"):
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            # MACD filter
            macd_options = [
                "Any MACD",
                "MACD above Signal Line",
                "MACD below Signal Line",
                "MACD Crossover (MACD crossing above Signal)",
                "MACD Crossunder (MACD crossing below Signal)"
            ]
            macd_filter = st.selectbox("MACD Filter", macd_options)
            
            # Bollinger Bands filter
            bb_options = [
                "Any Bollinger Bands Position",
                "Price above Upper Band",
                "Price below Lower Band",
                "Price near Upper Band",
                "Price near Lower Band",
                "Price in Middle Band"
            ]
            bb_filter = st.selectbox("Bollinger Bands Filter", bb_options)
        
        with adv_col2:
            # Support/Resistance filter
            sr_options = [
                "Any S/R Relationship",
                "Near Support Level",
                "Near Resistance Level",
                "Breakout above Resistance",
                "Breakdown below Support"
            ]
            sr_filter = st.selectbox("Support/Resistance Filter", sr_options)
            
            # Gap filter
            gap_options = [
                "No Gap Filter",
                "Gap Up",
                "Gap Down",
                "Gap Up (Filled)",
                "Gap Down (Filled)"
            ]
            gap_filter = st.selectbox("Gap Filter", gap_options)
    
    # Stock universe selection
    universe_options = ["S&P 500", "Dow 30", "Nasdaq 100", "Russell 1000", "Custom List"]
    stock_universe = st.selectbox("Stock Universe", universe_options)
    
    if stock_universe == "Custom List":
        custom_symbols = st.text_input("Enter comma-separated stock symbols (e.g., AAPL, MSFT, GOOGL)")
        symbols_list = [symbol.strip().upper() for symbol in custom_symbols.split(',')] if custom_symbols else []
    else:
        symbols_list = get_index_components(stock_universe)
    
    # Run screener
    if st.button("Run Technical Screener"):
        if not symbols_list:
            st.warning("Please select a stock universe or enter custom stock symbols.")
            return
        
        # Run the screener
        with st.spinner(f"Screening {len(symbols_list)} stocks..."):
            results = run_technical_screener(
                symbols_list,
                days,
                (min_performance, max_performance),
                (min_volatility, max_volatility),
                min_avg_volume,
                ma_filter,
                (min_rsi, max_rsi),
                trend_filter,
                macd_filter,
                bb_filter,
                sr_filter,
                gap_filter
            )
            
            display_screener_results(results, show_technical=True)

def show_strategy_screener():
    """Display pre-built strategy screener options"""
    # Define pre-built strategies
    strategies = {
        "Dividend Income": "Stocks with high dividend yields and strong financials",
        "Growth Leaders": "Fast-growing companies with strong revenue and earnings growth",
        "Value Opportunities": "Undervalued stocks with strong fundamentals",
        "Momentum Stars": "Stocks showing strong price momentum and technical strength",
        "Low Volatility": "Stable stocks with lower price fluctuations",
        "Beaten Down Bargains": "Stocks that have fallen significantly but show signs of recovery",
        "CANSLIM": "William O'Neil's CANSLIM growth strategy",
        "Warren Buffett Style": "Value companies with competitive advantages and quality management",
        "High Quality": "Companies with strong balance sheets and consistent profitability"
    }
    
    # Strategy selection
    selected_strategy = st.selectbox(
        "Select a pre-built investment strategy:", 
        list(strategies.keys())
    )
    
    # Display strategy description
    st.info(f"**Strategy Description:** {strategies[selected_strategy]}")
    
    # Stock universe selection
    universe_options = ["S&P 500", "Dow 30", "Nasdaq 100", "Russell 1000", "Custom List"]
    stock_universe = st.selectbox("Stock Universe", universe_options)
    
    if stock_universe == "Custom List":
        custom_symbols = st.text_input("Enter comma-separated stock symbols (e.g., AAPL, MSFT, GOOGL)")
        symbols_list = [symbol.strip().upper() for symbol in custom_symbols.split(',')] if custom_symbols else []
    else:
        symbols_list = get_index_components(stock_universe)
    
    # Number of stocks to return
    max_stocks = st.slider("Maximum number of stocks to return", 5, 50, 20)
    
    # Run screener
    if st.button("Run Strategy Screener"):
        if not symbols_list:
            st.warning("Please select a stock universe or enter custom stock symbols.")
            return
        
        # Run the screener
        with st.spinner(f"Screening {len(symbols_list)} stocks using {selected_strategy} strategy..."):
            results = run_strategy_screener(symbols_list, selected_strategy, max_stocks)
            
            if results is not None:
                # Check if strategy returns technical or fundamental focus
                if selected_strategy in ["Momentum Stars", "Beaten Down Bargains", "CANSLIM"]:
                    display_screener_results(results, show_technical=True, strategy_name=selected_strategy)
                else:
                    display_screener_results(results, strategy_name=selected_strategy)
            else:
                st.error("No stocks passed the screening criteria. Try adjusting your parameters.")

def run_fundamental_screener(symbols, sectors, market_cap, pe_range, dividend_range, 
                           price_range, min_revenue_growth, profitable_only, analyst_rating,
                           debt_to_equity_range, min_roe, profit_margin_range, min_peg):
    """Run fundamental screening on the provided symbols"""
    # Implement logic to screen stocks based on fundamental criteria
    results = []
    count = 0
    total = len(symbols)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for symbol in symbols:
        # Update progress
        count += 1
        progress = count / total
        progress_bar.progress(progress)
        status_text.text(f"Processing {count}/{total}: {symbol}")
        
        try:
            # Get company info
            company_info = data_fetcher.get_company_info(symbol)
            
            if company_info is None:
                continue
            
            # Check if sector matches
            sector = company_info.get('sector', '')
            if "All Sectors" not in sectors and sector not in sectors:
                continue
            
            # Check market cap
            market_cap_value = company_info.get('marketCap', 0)
            if market_cap != "Any Market Cap":
                if market_cap == "Mega Cap (> $200B)" and market_cap_value < 2e11:
                    continue
                elif market_cap == "Large Cap ($10B - $200B)" and (market_cap_value < 1e10 or market_cap_value > 2e11):
                    continue
                elif market_cap == "Mid Cap ($2B - $10B)" and (market_cap_value < 2e9 or market_cap_value > 1e10):
                    continue
                elif market_cap == "Small Cap ($300M - $2B)" and (market_cap_value < 3e8 or market_cap_value > 2e9):
                    continue
                elif market_cap == "Micro Cap (< $300M)" and market_cap_value > 3e8:
                    continue
            
            # Check P/E ratio
            pe_ratio = company_info.get('trailingPE')
            if pe_ratio is not None and (pe_ratio < pe_range[0] or pe_ratio > pe_range[1]):
                continue
            
            # Check dividend yield
            dividend_yield = company_info.get('dividendYield', 0)
            if dividend_yield is not None:
                dividend_pct = dividend_yield * 100
                if dividend_pct < dividend_range[0] or dividend_pct > dividend_range[1]:
                    continue
            
            # Get current price data
            current_price = data_fetcher.get_current_price(symbol)
            if current_price is None or current_price < price_range[0] or current_price > price_range[1]:
                continue
            
            # Check revenue growth
            revenue_growth = company_info.get('revenueGrowth', 0)
            if revenue_growth is not None and revenue_growth * 100 < min_revenue_growth:
                continue
            
            # Check if profitable
            if profitable_only and (company_info.get('netIncomeToCommon', 0) <= 0):
                continue
            
            # Check analyst rating
            rating = company_info.get('recommendationKey', 'hold')
            rating_map = {'strongBuy': 4, 'buy': 3, 'hold': 2, 'sell': 1, 'strongSell': 0}
            if analyst_rating != "Any Rating":
                rating_value = rating_map.get(rating, 2)
                min_rating_value = rating_map.get(analyst_rating.lower().replace(' ', ''), 0)
                if rating_value < min_rating_value:
                    continue
            
            # Check advanced criteria
            debt_to_equity = company_info.get('debtToEquity')
            if debt_to_equity is not None and (debt_to_equity < debt_to_equity_range[0] or 
                                              debt_to_equity > debt_to_equity_range[1]):
                continue
            
            roe = company_info.get('returnOnEquity', 0)
            if roe is not None and roe * 100 < min_roe:
                continue
            
            profit_margin = company_info.get('profitMargins', 0)
            if profit_margin is not None and (profit_margin * 100 < profit_margin_range[0] or 
                                             profit_margin * 100 > profit_margin_range[1]):
                continue
            
            peg_ratio = company_info.get('pegRatio')
            if peg_ratio is not None and min_peg > 0 and peg_ratio > min_peg:
                continue
            
            # Stock passed all filters, add to results
            results.append({
                'Symbol': symbol,
                'Name': company_info.get('shortName', symbol),
                'Sector': sector,
                'Industry': company_info.get('industry', 'N/A'),
                'Price': current_price,
                'Market Cap ($B)': market_cap_value / 1e9,
                'P/E Ratio': pe_ratio,
                'Dividend Yield (%)': dividend_yield * 100 if dividend_yield else 0,
                'Revenue Growth (%)': revenue_growth * 100 if revenue_growth else 0,
                'Profit Margin (%)': profit_margin * 100 if profit_margin else 0,
                'ROE (%)': roe * 100 if roe else 0,
                'Debt to Equity': debt_to_equity,
                'PEG Ratio': peg_ratio,
                'Analyst Rating': rating.capitalize() if rating else 'N/A'
            })
        
        except Exception as e:
            st.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def run_technical_screener(symbols, days, performance_range, volatility_range, 
                         min_avg_volume, ma_filter, rsi_range, trend_filter,
                         macd_filter, bb_filter, sr_filter, gap_filter):
    """Run technical screening on the provided symbols"""
    # Implement logic to screen stocks based on technical criteria
    results = []
    count = 0
    total = len(symbols)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for symbol in symbols:
        # Update progress
        count += 1
        progress = count / total
        progress_bar.progress(progress)
        status_text.text(f"Processing {count}/{total}: {symbol}")
        
        try:
            # Get historical stock data
            stock_data = data_fetcher.get_stock_data(symbol, period=f"{days+200}d")  # Add extra days for MA calculation
            
            if stock_data is None or stock_data.empty:
                continue
            
            # Calculate start date
            start_date = pd.Timestamp(datetime.now() - timedelta(days=days)).tz_localize(None)
            
            # Handle timezone-aware index properly
            if stock_data.index.tz is not None:
                compare_index = stock_data.index.tz_localize(None)
                filtered_data = stock_data[compare_index >= start_date]
            else:
                filtered_data = stock_data[stock_data.index >= start_date]
            
            if filtered_data.empty or len(filtered_data) < 20:
                continue
            
            # Get company info
            company_info = data_fetcher.get_company_info(symbol)
            if company_info is None:
                continue
            
            # Calculate performance
            start_price = filtered_data['Close'].iloc[0]
            end_price = filtered_data['Close'].iloc[-1]
            performance = ((end_price / start_price) - 1) * 100
            
            if performance < performance_range[0] or performance > performance_range[1]:
                continue
            
            # Calculate volatility
            returns = filtered_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            if volatility < volatility_range[0] or volatility > volatility_range[1]:
                continue
            
            # Check volume
            avg_volume = filtered_data['Volume'].mean()
            if avg_volume < min_avg_volume:
                continue
            
            # Calculate moving averages
            if len(stock_data) >= 200:
                stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
                stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
                
                if ma_filter != "Any MA Relationship":
                    current_price = stock_data['Close'].iloc[-1]
                    current_ma50 = stock_data['MA50'].iloc[-1]
                    current_ma200 = stock_data['MA200'].iloc[-1]
                    
                    # Check specific MA relationships
                    if ma_filter == "Price above 50-day MA" and current_price <= current_ma50:
                        continue
                    elif ma_filter == "Price below 50-day MA" and current_price >= current_ma50:
                        continue
                    elif ma_filter == "Price above 200-day MA" and current_price <= current_ma200:
                        continue
                    elif ma_filter == "Price below 200-day MA" and current_price >= current_ma200:
                        continue
                    elif ma_filter == "50-day MA above 200-day MA" and current_ma50 <= current_ma200:
                        continue
                    elif ma_filter == "50-day MA below 200-day MA" and current_ma50 >= current_ma200:
                        continue
                    
                    # Check for crosses
                    elif ma_filter == "Golden Cross (50-day crossing above 200-day)":
                        # Check last 10 days for cross
                        recent_data = stock_data.iloc[-10:]
                        had_cross = False
                        for i in range(1, len(recent_data)):
                            if (recent_data['MA50'].iloc[i-1] <= recent_data['MA200'].iloc[i-1] and
                                recent_data['MA50'].iloc[i] > recent_data['MA200'].iloc[i]):
                                had_cross = True
                                break
                        if not had_cross:
                            continue
                    
                    elif ma_filter == "Death Cross (50-day crossing below 200-day)":
                        # Check last 10 days for cross
                        recent_data = stock_data.iloc[-10:]
                        had_cross = False
                        for i in range(1, len(recent_data)):
                            if (recent_data['MA50'].iloc[i-1] >= recent_data['MA200'].iloc[i-1] and
                                recent_data['MA50'].iloc[i] < recent_data['MA200'].iloc[i]):
                                had_cross = True
                                break
                        if not had_cross:
                            continue
            
            # Calculate RSI
            def calculate_rsi(data, window=14):
                delta = data.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                
                avg_gain = gain.rolling(window=window).mean()
                avg_loss = loss.rolling(window=window).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            filtered_data['RSI'] = calculate_rsi(filtered_data['Close'])
            current_rsi = filtered_data['RSI'].iloc[-1]
            
            if current_rsi < rsi_range[0] or current_rsi > rsi_range[1]:
                continue
            
            # Check trend
            if trend_filter != "Any Trend":
                # Calculate trend using simple linear regression slope
                x = np.arange(len(filtered_data))
                y = filtered_data['Close'].values
                slope = np.polyfit(x, y, 1)[0]
                
                if (trend_filter == "Uptrend" and slope <= 0) or (trend_filter == "Downtrend" and slope >= 0):
                    continue
                elif trend_filter == "Sideways" and (abs(slope) > 0.1 * np.mean(y) / len(y)):
                    continue
            
            # Calculate MACD
            if macd_filter != "Any MACD":
                # Calculate MACD
                filtered_data['EMA12'] = filtered_data['Close'].ewm(span=12, adjust=False).mean()
                filtered_data['EMA26'] = filtered_data['Close'].ewm(span=26, adjust=False).mean()
                filtered_data['MACD'] = filtered_data['EMA12'] - filtered_data['EMA26']
                filtered_data['Signal'] = filtered_data['MACD'].ewm(span=9, adjust=False).mean()
                
                current_macd = filtered_data['MACD'].iloc[-1]
                current_signal = filtered_data['Signal'].iloc[-1]
                
                if macd_filter == "MACD above Signal Line" and current_macd <= current_signal:
                    continue
                elif macd_filter == "MACD below Signal Line" and current_macd >= current_signal:
                    continue
                elif macd_filter == "MACD Crossover (MACD crossing above Signal)":
                    # Check last 3 days for cross
                    recent_data = filtered_data.iloc[-5:]
                    had_cross = False
                    for i in range(1, len(recent_data)):
                        if (recent_data['MACD'].iloc[i-1] <= recent_data['Signal'].iloc[i-1] and
                            recent_data['MACD'].iloc[i] > recent_data['Signal'].iloc[i]):
                            had_cross = True
                            break
                    if not had_cross:
                        continue
                elif macd_filter == "MACD Crossunder (MACD crossing below Signal)":
                    # Check last 3 days for cross
                    recent_data = filtered_data.iloc[-5:]
                    had_cross = False
                    for i in range(1, len(recent_data)):
                        if (recent_data['MACD'].iloc[i-1] >= recent_data['Signal'].iloc[i-1] and
                            recent_data['MACD'].iloc[i] < recent_data['Signal'].iloc[i]):
                            had_cross = True
                            break
                    if not had_cross:
                        continue
            
            # Calculate Bollinger Bands
            if bb_filter != "Any Bollinger Bands Position":
                # Calculate Bollinger Bands
                filtered_data['MA20'] = filtered_data['Close'].rolling(window=20).mean()
                filtered_data['STD20'] = filtered_data['Close'].rolling(window=20).std()
                filtered_data['UpperBand'] = filtered_data['MA20'] + (filtered_data['STD20'] * 2)
                filtered_data['LowerBand'] = filtered_data['MA20'] - (filtered_data['STD20'] * 2)
                
                current_price = filtered_data['Close'].iloc[-1]
                current_upper = filtered_data['UpperBand'].iloc[-1]
                current_lower = filtered_data['LowerBand'].iloc[-1]
                current_middle = filtered_data['MA20'].iloc[-1]
                
                if bb_filter == "Price above Upper Band" and current_price <= current_upper:
                    continue
                elif bb_filter == "Price below Lower Band" and current_price >= current_lower:
                    continue
                elif bb_filter == "Price near Upper Band" and (current_price < current_upper * 0.95):
                    continue
                elif bb_filter == "Price near Lower Band" and (current_price > current_lower * 1.05):
                    continue
                elif bb_filter == "Price in Middle Band" and (current_price < current_middle * 0.95 or 
                                                           current_price > current_middle * 1.05):
                    continue
            
            # Skip the more complex filters for simplicity in this demo
            # In a real application, support/resistance and gap detection would require more complex algorithms
            
            # Stock passed all filters, add to results
            results.append({
                'Symbol': symbol,
                'Name': company_info.get('shortName', symbol),
                'Sector': company_info.get('sector', 'N/A'),
                'Price': filtered_data['Close'].iloc[-1],
                'Performance (%)': performance,
                'Volatility (%)': volatility,
                'RSI': current_rsi,
                'Avg Volume': avg_volume,
                'Market Cap ($B)': company_info.get('marketCap', 0) / 1e9,
                'MA50': stock_data['MA50'].iloc[-1] if 'MA50' in stock_data.columns else None,
                'MA200': stock_data['MA200'].iloc[-1] if 'MA200' in stock_data.columns else None,
                'MA Relationship': 'Above 200-day' if 'MA200' in stock_data.columns and filtered_data['Close'].iloc[-1] > stock_data['MA200'].iloc[-1] else 'Below 200-day'
            })
        
        except Exception as e:
            st.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return results

def run_strategy_screener(symbols, strategy, max_stocks):
    """Run screening based on a pre-built strategy"""
    if strategy == "Dividend Income":
        # High dividend yield, stable companies, strong financials
        results = run_fundamental_screener(
            symbols=symbols,
            sectors=["All Sectors"],
            market_cap="Large Cap ($10B - $200B)",
            pe_range=(0, 30),
            dividend_range=(3.0, 15.0),  # Focus on high dividend yield
            price_range=(5.0, 1000.0),
            min_revenue_growth=0.0,
            profitable_only=True,
            analyst_rating="Hold",
            debt_to_equity_range=(0.0, 1.5),  # Low debt
            min_roe=8.0,  # Decent ROE
            profit_margin_range=(5.0, 50.0),
            min_peg=0.0
        )
    
    elif strategy == "Growth Leaders":
        # Fast-growing companies with strong revenue growth
        results = run_fundamental_screener(
            symbols=symbols,
            sectors=["All Sectors"],
            market_cap="Any Market Cap",
            pe_range=(0, 100),
            dividend_range=(0.0, 3.0),  # Growth companies often have low dividends
            price_range=(10.0, 1000.0),
            min_revenue_growth=15.0,  # High revenue growth
            profitable_only=True,
            analyst_rating="Buy",
            debt_to_equity_range=(0.0, 2.0),
            min_roe=10.0,
            profit_margin_range=(5.0, 50.0),
            min_peg=0.0
        )
    
    elif strategy == "Value Opportunities":
        # Undervalued stocks with good fundamentals
        results = run_fundamental_screener(
            symbols=symbols,
            sectors=["All Sectors"],
            market_cap="Any Market Cap",
            pe_range=(0, 15),  # Low P/E ratio
            dividend_range=(0.0, 20.0),
            price_range=(5.0, 100.0),  # Lower price range
            min_revenue_growth=0.0,
            profitable_only=True,
            analyst_rating="Hold",
            debt_to_equity_range=(0.0, 1.0),
            min_roe=5.0,
            profit_margin_range=(3.0, 50.0),
            min_peg=2.0  # Lower PEG ratio
        )
    
    elif strategy == "Momentum Stars":
        # Stocks with strong price momentum
        results = run_technical_screener(
            symbols=symbols,
            days=90,  # 3 months lookback
            performance_range=(15.0, 100.0),  # Strong recent performance
            volatility_range=(0.0, 100.0),
            min_avg_volume=500000,
            ma_filter="Price above 50-day MA",
            rsi_range=(50, 80),  # Strong RSI but not overbought
            trend_filter="Uptrend",
            macd_filter="MACD above Signal Line",
            bb_filter="Any Bollinger Bands Position",
            sr_filter="Any S/R Relationship",
            gap_filter="No Gap Filter"
        )
    
    elif strategy == "Low Volatility":
        # Stable stocks with lower volatility
        results = run_technical_screener(
            symbols=symbols,
            days=180,  # 6 months lookback
            performance_range=(-10.0, 30.0),
            volatility_range=(0.0, 20.0),  # Low volatility
            min_avg_volume=200000,
            ma_filter="Any MA Relationship",
            rsi_range=(40, 60),  # Neutral RSI
            trend_filter="Any Trend",
            macd_filter="Any MACD",
            bb_filter="Price in Middle Band",
            sr_filter="Any S/R Relationship",
            gap_filter="No Gap Filter"
        )
    
    elif strategy == "Beaten Down Bargains":
        # Stocks that have fallen significantly but show signs of recovery
        results = run_technical_screener(
            symbols=symbols,
            days=180,  # 6 months lookback
            performance_range=(-50.0, -10.0),  # Significant recent decline
            volatility_range=(0.0, 100.0),
            min_avg_volume=200000,
            ma_filter="Any MA Relationship",
            rsi_range=(30, 50),  # RSI starting to rise from oversold
            trend_filter="Any Trend",
            macd_filter="MACD Crossover (MACD crossing above Signal)",
            bb_filter="Price near Lower Band",
            sr_filter="Any S/R Relationship",
            gap_filter="No Gap Filter"
        )
    
    elif strategy == "CANSLIM":
        # William O'Neil's CANSLIM growth strategy
        results = run_technical_screener(
            symbols=symbols,
            days=90,  # 3 months lookback
            performance_range=(10.0, 100.0),  # Strong recent performance
            volatility_range=(0.0, 100.0),
            min_avg_volume=400000,  # Good institutional support
            ma_filter="Price above 200-day MA",  # Uptrend
            rsi_range=(0, 100),
            trend_filter="Uptrend",
            macd_filter="Any MACD",
            bb_filter="Any Bollinger Bands Position",
            sr_filter="Any S/R Relationship",
            gap_filter="No Gap Filter"
        )
        
        # Further refine to match CANSLIM criteria if we had more data
        # (would typically look at quarterly earnings growth, etc.)
    
    elif strategy == "Warren Buffett Style":
        # Value companies with competitive advantages
        results = run_fundamental_screener(
            symbols=symbols,
            sectors=["All Sectors"],
            market_cap="Large Cap ($10B - $200B)",
            pe_range=(0, 25),  # Reasonable P/E
            dividend_range=(0.0, 20.0),
            price_range=(0.0, 1000.0),
            min_revenue_growth=5.0,  # Consistent growth
            profitable_only=True,
            analyst_rating="Any Rating",
            debt_to_equity_range=(0.0, 0.5),  # Low debt
            min_roe=15.0,  # High ROE indicates competitive advantage
            profit_margin_range=(15.0, 50.0),  # High margins
            min_peg=0.0
        )
    
    else:  # High Quality
        # Companies with strong balance sheets and consistent profitability
        results = run_fundamental_screener(
            symbols=symbols,
            sectors=["All Sectors"],
            market_cap="Any Market Cap",
            pe_range=(0, 30),
            dividend_range=(0.0, 20.0),
            price_range=(0.0, 1000.0),
            min_revenue_growth=3.0,
            profitable_only=True,
            analyst_rating="Any Rating",
            debt_to_equity_range=(0.0, 0.7),  # Low debt
            min_roe=12.0,  # Good ROE
            profit_margin_range=(10.0, 50.0),  # Good margins
            min_peg=0.0
        )
    
    # Limit results
    if results and len(results) > max_stocks:
        # Sort based on strategy
        if strategy == "Dividend Income":
            results = sorted(results, key=lambda x: x.get('Dividend Yield (%)', 0), reverse=True)
        elif strategy == "Growth Leaders":
            results = sorted(results, key=lambda x: x.get('Revenue Growth (%)', 0), reverse=True)
        elif strategy == "Value Opportunities":
            results = sorted(results, key=lambda x: x.get('P/E Ratio', float('inf')))
        elif strategy == "Momentum Stars":
            results = sorted(results, key=lambda x: x.get('Performance (%)', 0), reverse=True)
        elif strategy == "Low Volatility":
            results = sorted(results, key=lambda x: x.get('Volatility (%)', float('inf')))
        elif strategy == "Beaten Down Bargains":
            results = sorted(results, key=lambda x: x.get('RSI', 0))
        elif strategy == "CANSLIM":
            results = sorted(results, key=lambda x: x.get('Performance (%)', 0), reverse=True)
        elif strategy == "Warren Buffett Style":
            results = sorted(results, key=lambda x: x.get('ROE (%)', 0), reverse=True)
        else:  # High Quality
            results = sorted(results, key=lambda x: x.get('Profit Margin (%)', 0), reverse=True)
        
        results = results[:max_stocks]
    
    return results

def display_screener_results(results, show_technical=False, strategy_name=None):
    """Display the screening results to the user"""
    if not results:
        st.warning("No stocks passed the screening criteria. Try adjusting your parameters.")
        return
    
    # Display number of results
    st.success(f"Found {len(results)} stocks matching your criteria.")
    
    # Create dataframe from results
    df = pd.DataFrame(results)
    
    # Round numeric columns
    numeric_columns = []
    for col in df.columns:
        if col not in ['Symbol', 'Name', 'Sector', 'Industry', 'MA Relationship', 'Analyst Rating']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_columns.append(col)
            except:
                pass
    
    if numeric_columns:
        df[numeric_columns] = df[numeric_columns].round(2)
    
    # Format average volume
    if 'Avg Volume' in df.columns:
        df['Avg Volume'] = df['Avg Volume'].apply(lambda x: f"{int(x):,}")
    
    # Sort by column
    if show_technical:
        sort_col = 'Performance (%)' if 'Performance (%)' in df.columns else 'Symbol'
        df = df.sort_values(sort_col, ascending=False)
    else:
        sort_col = 'Market Cap ($B)' if 'Market Cap ($B)' in df.columns else 'Symbol'
        df = df.sort_values(sort_col, ascending=False)
    
    # Display dataframe with formatting
    def highlight_performance(val):
        if isinstance(val, (float, int)) or (isinstance(val, str) and val.replace('.', '').replace('-', '').isdigit()):
            val_float = float(val)
            if val_float > 0:
                return 'color: green'
            elif val_float < 0:
                return 'color: red'
        return ''
    
    # Apply highlighting
    if show_technical and 'Performance (%)' in df.columns:
        styled_df = df.style.applymap(highlight_performance, subset=['Performance (%)'])
    elif 'Revenue Growth (%)' in df.columns:
        styled_df = df.style.applymap(highlight_performance, subset=['Revenue Growth (%)'])
    else:
        styled_df = df.style
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Add strategy-specific insights
    if strategy_name:
        st.subheader(f"{strategy_name} Strategy Insights")
        
        if strategy_name == "Dividend Income":
            st.info("""
            **Dividend Income Strategy**
            
            These stocks offer higher dividend yields with generally stable business models. 
            Key considerations:
            - Focus on dividend sustainability and growth history
            - Watch for dividend payout ratios (higher can be less sustainable)
            - Diversify across different sectors for income stability
            """)
        
        elif strategy_name == "Growth Leaders":
            st.info("""
            **Growth Leaders Strategy**
            
            These companies show strong revenue and earnings growth trajectories.
            Key considerations:
            - Growth stocks can be more volatile
            - Valuations may be higher than market averages
            - Pay attention to future growth expectations vs. historical growth
            """)
        
        elif strategy_name == "Value Opportunities":
            st.info("""
            **Value Opportunities Strategy**
            
            These stocks appear undervalued relative to their fundamentals.
            Key considerations:
            - Determine if low valuation is justified or a potential opportunity
            - Check for catalysts that might unlock value
            - Value stocks can underperform in strong bull markets
            """)
        
        elif strategy_name == "Momentum Stars":
            st.info("""
            **Momentum Stars Strategy**
            
            These stocks show strong price momentum and technical strength.
            Key considerations:
            - Momentum strategies require more active management
            - Set stop losses to protect against reversals
            - Monitor earnings dates and other potential volatility events
            """)
    
    # Visualization options
    st.subheader("Visualization")
    
    visualization_type = st.radio(
        "Select visualization type:",
        ["Sector Distribution", "Performance Comparison", "Scatter Plot Analysis"],
        horizontal=True
    )
    
    if visualization_type == "Sector Distribution":
        # Create sector distribution chart
        if 'Sector' in df.columns:
            sector_counts = df['Sector'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=sector_counts.index,
                values=sector_counts.values,
                hole=.3,
                marker_colors=px.colors.qualitative.Pastel
            )])
            
            fig.update_layout(
                title="Sector Distribution of Screened Stocks",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sector data not available for visualization.")
    
    elif visualization_type == "Performance Comparison":
        # Performance comparison (different for technical vs fundamental)
        if show_technical and 'Performance (%)' in df.columns:
            # Sort by performance
            sorted_df = df.sort_values('Performance (%)', ascending=False)
            
            fig = go.Figure(data=[go.Bar(
                x=sorted_df['Symbol'],
                y=sorted_df['Performance (%)'],
                marker_color=['green' if p > 0 else 'red' for p in sorted_df['Performance (%)']]
            )])
            
            fig.update_layout(
                title="Performance Comparison",
                xaxis_title="Stock Symbol",
                yaxis_title="Performance (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif 'Dividend Yield (%)' in df.columns:
            # Sort by dividend yield
            sorted_df = df.sort_values('Dividend Yield (%)', ascending=False)
            
            fig = go.Figure(data=[go.Bar(
                x=sorted_df['Symbol'],
                y=sorted_df['Dividend Yield (%)'],
                marker_color='royalblue'
            )])
            
            fig.update_layout(
                title="Dividend Yield Comparison",
                xaxis_title="Stock Symbol",
                yaxis_title="Dividend Yield (%)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Performance data not available for visualization.")
    
    else:  # Scatter Plot
        # Create scatter plot
        if show_technical and 'RSI' in df.columns and 'Performance (%)' in df.columns:
            # Technical scatter plot
            fig = go.Figure(data=go.Scatter(
                x=df['RSI'],
                y=df['Performance (%)'],
                mode='markers+text',
                text=df['Symbol'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=df['Volatility (%)'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Volatility (%)")
                )
            ))
            
            fig.update_layout(
                title="Performance vs RSI",
                xaxis_title="RSI",
                yaxis_title="Performance (%)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif 'P/E Ratio' in df.columns and 'Market Cap ($B)' in df.columns:
            # Fundamental scatter plot
            fig = go.Figure(data=go.Scatter(
                x=df['P/E Ratio'],
                y=df['Market Cap ($B)'],
                mode='markers+text',
                text=df['Symbol'],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=df['Dividend Yield (%)'] if 'Dividend Yield (%)' in df.columns else None,
                    colorscale='Viridis',
                    showscale=True if 'Dividend Yield (%)' in df.columns else False,
                    colorbar=dict(title="Dividend Yield (%)") if 'Dividend Yield (%)' in df.columns else None
                )
            ))
            
            fig.update_layout(
                title="Market Cap vs P/E Ratio",
                xaxis_title="P/E Ratio",
                yaxis_title="Market Cap ($B)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Data not available for scatter plot visualization.")

def get_index_components(index_name):
    """Get components of a major index"""
    # Simplified implementation - in a real app, this would fetch actual components
    if index_name == "Dow 30":
        return ["AAPL", "MSFT", "WMT", "HD", "AMGN", "BA", "CAT", "CSCO", "CVX", "GS", 
                "JNJ", "JPM", "MCD", "MRK", "NKE", "PG", "TRV", "UNH", "CRM", "VZ"]
    elif index_name == "S&P 500":
        # Return a subset of S&P 500 for demo purposes
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "JNJ", "V", "PG", 
                "HD", "BAC", "MA", "DIS", "NVDA", "PYPL", "NFLX", "INTC", "VZ", "ADBE",
                "WMT", "KO", "PEP", "MRK", "PFE", "T", "ABT", "CSCO", "CMCSA", "XOM"]
    elif index_name == "Nasdaq 100":
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "PYPL", "NFLX", "INTC", 
                "ADBE", "CMCSA", "PEP", "COST", "AMGN", "AVGO", "TXN", "CHTR", "SBUX", "QCOM"]
    else:  # Russell 1000
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "JNJ", "V", "PG", 
                "HD", "BAC", "MA", "DIS", "NVDA", "PYPL", "NFLX", "INTC", "VZ", "ADBE"]