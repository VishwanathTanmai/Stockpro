import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import database
import data_fetcher

def show_trading_page():
    """Display the trading interface"""
    if not st.session_state.logged_in:
        st.warning("Please login to access trading features.")
        return
    
    st.header("Trading Platform")
    
    # Get user's balance and portfolio
    username = st.session_state.username
    user_data = database.get_user_data(username)
    balance = user_data.get('balance', 10000.0)
    portfolio = user_data.get('portfolio', {})
    
    # Display available balance
    st.metric("Available Cash", f"${balance:.2f}")
    
    # Trade selection tabs
    tab1, tab2 = st.tabs(["Buy/Sell Stocks", "Order History"])
    
    with tab1:
        # Search for a stock
        st.subheader("Search for a Stock")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbol = st.text_input("Enter stock symbol (e.g. AAPL, MSFT, GOOGL)", key="trading_symbol")
        
        with col2:
            if st.button("Search", key="search_stock_button"):
                if symbol:
                    st.session_state.search_symbol = symbol.upper()
                else:
                    st.warning("Please enter a stock symbol.")
        
        # Display stock information if searched
        if 'search_symbol' in st.session_state and st.session_state.search_symbol:
            symbol = st.session_state.search_symbol
            stock_data = data_fetcher.get_stock_data(symbol, period="5d")
            
            if stock_data is not None:
                # Show basic stock info
                current_price = stock_data['Close'].iloc[-1]
                prev_close = stock_data['Close'].iloc[-2]
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
                
                # Company info
                company_info = data_fetcher.get_company_info(symbol)
                company_name = company_info.get('longName', symbol)
                
                # Display stock header
                st.subheader(f"{company_name} ({symbol})")
                
                # Price and change
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    price_color = "green" if price_change >= 0 else "red"
                    change_icon = "▲" if price_change >= 0 else "▼"
                    
                    st.markdown(f"""
                    <div>
                        <span style='font-size: 1.5rem; font-weight: bold;'>${current_price:.2f}</span>
                        <span style='font-size: 1.1rem; color: {price_color};'> {change_icon} {abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Current position if any
                    if symbol in portfolio:
                        position = portfolio[symbol]
                        position_qty = position['quantity']
                        position_avg = position['avg_price']
                        position_value = position_qty * current_price
                        position_cost = position_qty * position_avg
                        position_pl = position_value - position_cost
                        position_pl_pct = (position_pl / position_cost) * 100 if position_cost > 0 else 0
                        
                        st.markdown("**Your Position:**")
                        st.write(f"Shares: {position_qty}")
                        st.write(f"Avg Price: ${position_avg:.2f}")
                        
                        pl_color = "green" if position_pl >= 0 else "red"
                        st.markdown(f"P/L: <span style='color: {pl_color};'>${position_pl:.2f} ({position_pl_pct:.2f}%)</span>", unsafe_allow_html=True)
                
                with col2:
                    # Stock chart for last 5 days
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='royalblue', width=2)
                    ))
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis_title='Price (USD)',
                        xaxis_title=None,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trading interface
                st.subheader("Place Order")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Buy form
                    st.markdown("### Buy")
                    buy_quantity = st.number_input("Quantity to Buy", min_value=1, value=1, step=1, key="buy_quantity")
                    buy_total = buy_quantity * current_price
                    
                    st.write(f"Total Cost: ${buy_total:.2f}")
                    
                    if st.button("Buy", key="buy_button"):
                        if buy_total > balance:
                            st.error("Insufficient funds for this purchase.")
                        else:
                            # Execute buy order
                            success = execute_order(username, symbol, company_name, buy_quantity, current_price, "buy")
                            if success:
                                st.success(f"Successfully purchased {buy_quantity} shares of {symbol}.")
                                # Update session state
                                st.session_state.balance = database.get_user_data(username)['balance']
                                st.rerun()
                            else:
                                st.error("Failed to execute buy order. Please try again.")
                
                with col2:
                    # Sell form
                    st.markdown("### Sell")
                    
                    # Check if user has shares to sell
                    max_sell = 0
                    if symbol in portfolio:
                        max_sell = portfolio[symbol]['quantity']
                    
                    if max_sell > 0:
                        sell_quantity = st.number_input("Quantity to Sell", min_value=1, max_value=max_sell, value=1, step=1, key="sell_quantity")
                        sell_total = sell_quantity * current_price
                        
                        st.write(f"Total Proceeds: ${sell_total:.2f}")
                        
                        if st.button("Sell", key="sell_button"):
                            # Execute sell order
                            success = execute_order(username, symbol, company_name, sell_quantity, current_price, "sell")
                            if success:
                                st.success(f"Successfully sold {sell_quantity} shares of {symbol}.")
                                # Update session state
                                st.session_state.balance = database.get_user_data(username)['balance']
                                st.rerun()
                            else:
                                st.error("Failed to execute sell order. Please try again.")
                    else:
                        st.write("You don't own any shares of this stock.")
            else:
                st.error(f"Could not find data for stock symbol: {symbol}")
    
    with tab2:
        # Show order history
        st.subheader("Order History")
        
        # Get trading history
        history_df = database.get_trading_history_as_df(username)
        
        if history_df is not None and not history_df.empty:
            # Format the dataframe
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp', ascending=False)
            
            # Format columns for display
            display_df = history_df.copy()
            display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
            display_df['total_amount'] = display_df['total_amount'].apply(lambda x: f"${x:.2f}")
            
            # Color code order types
            def highlight_order_type(val):
                if val == 'buy':
                    return 'color: green'
                elif val == 'sell':
                    return 'color: red'
                return ''
            
            st.dataframe(
                display_df.style.applymap(highlight_order_type, subset=['order_type']),
                use_container_width=True
            )
        else:
            st.info("No trading history yet.")

def execute_order(username, symbol, company_name, quantity, price, order_type):
    """Execute a buy or sell order"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_amount = quantity * price
    
    return database.record_trade(
        username=username,
        symbol=symbol,
        company_name=company_name,
        quantity=quantity,
        price=price,
        total_amount=total_amount,
        order_type=order_type,
        timestamp=timestamp
    )
