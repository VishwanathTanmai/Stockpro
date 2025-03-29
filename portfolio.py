import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import database
import data_fetcher

def show_portfolio():
    """Display the user's portfolio and its performance"""
    if not st.session_state.logged_in:
        st.warning("Please login to view your portfolio.")
        return
    
    st.header("Your Portfolio")
    
    # Get user data
    username = st.session_state.username
    user_data = database.get_user_data(username)
    portfolio = user_data.get('portfolio', {})
    balance = user_data.get('balance', 10000.0)
    
    # Show current balance
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Available Cash", f"${balance:.2f}")
    
    with col2:
        # Calculate portfolio value
        portfolio_value = calculate_portfolio_value(portfolio)
        total_value = balance + portfolio_value
        st.metric("Total Portfolio Value", f"${total_value:.2f}")
    
    # Portfolio breakdown
    st.subheader("Holdings")
    
    if not portfolio:
        st.info("Your portfolio is empty. Start trading to build your portfolio!")
    else:
        # Create a dataframe for the holdings
        holdings_data = []
        total_investment = 0
        total_current_value = 0
        
        for symbol, details in portfolio.items():
            current_price = data_fetcher.get_current_price(symbol)
            if current_price is None:
                continue
                
            quantity = details['quantity']
            avg_price = details['avg_price']
            cost_basis = quantity * avg_price
            current_value = quantity * current_price
            profit_loss = current_value - cost_basis
            profit_loss_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            total_investment += cost_basis
            total_current_value += current_value
            
            holdings_data.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Avg Price': avg_price,
                'Current Price': current_price,
                'Cost Basis': cost_basis,
                'Current Value': current_value,
                'Profit/Loss': profit_loss,
                'Profit/Loss %': profit_loss_pct
            })
        
        if holdings_data:
            # Calculate total profit/loss
            total_profit_loss = total_current_value - total_investment
            total_profit_loss_pct = (total_profit_loss / total_investment) * 100 if total_investment > 0 else 0
            
            # Add a row for totals
            holdings_data.append({
                'Symbol': 'TOTAL',
                'Quantity': '',
                'Avg Price': '',
                'Current Price': '',
                'Cost Basis': total_investment,
                'Current Value': total_current_value,
                'Profit/Loss': total_profit_loss,
                'Profit/Loss %': total_profit_loss_pct
            })
            
            holdings_df = pd.DataFrame(holdings_data)
            
            # Format the dataframe
            display_df = holdings_df.copy()
            display_df['Avg Price'] = display_df['Avg Price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else x)
            display_df['Cost Basis'] = display_df['Cost Basis'].apply(lambda x: f"${x:.2f}")
            display_df['Current Value'] = display_df['Current Value'].apply(lambda x: f"${x:.2f}")
            display_df['Profit/Loss'] = display_df['Profit/Loss'].apply(lambda x: f"${x:.2f}")
            display_df['Profit/Loss %'] = display_df['Profit/Loss %'].apply(lambda x: f"{x:.2f}%")
            
            # Style the dataframe
            def highlight_profit_loss(val):
                if not isinstance(val, str):
                    return ''
                if '$' in val:
                    try:
                        value = float(val.replace('$', ''))
                        if value > 0:
                            return 'color: green'
                        elif value < 0:
                            return 'color: red'
                    except:
                        pass
                elif '%' in val:
                    try:
                        value = float(val.replace('%', ''))
                        if value > 0:
                            return 'color: green'
                        elif value < 0:
                            return 'color: red'
                    except:
                        pass
                return ''
            
            # Display styled dataframe
            st.dataframe(
                display_df.style.applymap(
                    highlight_profit_loss, 
                    subset=['Profit/Loss', 'Profit/Loss %']
                ),
                use_container_width=True
            )
            
            # Portfolio Visualization
            st.subheader("Portfolio Allocation")
            
            # Exclude the TOTAL row for visualization
            viz_df = holdings_df[holdings_df['Symbol'] != 'TOTAL'].copy()
            
            # Create a pie chart for asset allocation
            if len(viz_df) > 0:
                fig = px.pie(
                    viz_df, 
                    names='Symbol', 
                    values='Current Value',
                    title='Portfolio Composition by Value',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    height=500,
                    margin=dict(t=40, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance by stock
                st.subheader("Performance by Stock")
                
                # Bar chart for profit/loss by stock
                fig_perf = px.bar(
                    viz_df,
                    x='Symbol',
                    y='Profit/Loss',
                    color='Profit/Loss',
                    color_continuous_scale=['red', 'green'],
                    title='Profit/Loss by Stock ($)'
                )
                
                fig_perf.update_layout(
                    height=400,
                    margin=dict(t=40, b=0, l=0, r=0),
                    coloraxis_showscale=False
                )
                
                st.plotly_chart(fig_perf, use_container_width=True)
        else:
            st.info("Failed to retrieve current market data for your portfolio.")
    
    # Portfolio performance over time if available
    show_portfolio_performance(username)

def calculate_portfolio_value(portfolio):
    """Calculate the current value of the portfolio"""
    total_value = 0
    for symbol, details in portfolio.items():
        current_price = data_fetcher.get_current_price(symbol)
        if current_price:
            total_value += details['quantity'] * current_price
    return total_value

def show_portfolio_performance(username):
    """Show the portfolio performance over time if history is available"""
    history = database.get_portfolio_history(username)
    
    if history and len(history) > 1:
        st.subheader("Portfolio Performance")
        
        # Create dataframe for portfolio history
        history_df = pd.DataFrame(history)
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.sort_values('date')
        
        # Create a line chart for portfolio performance
        fig = px.line(
            history_df, 
            x='date', 
            y='total_value',
            title='Portfolio Value Over Time',
            labels={'date': 'Date', 'total_value': 'Total Value ($)'}
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=40, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data to display performance chart yet.")
