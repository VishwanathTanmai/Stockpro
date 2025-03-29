import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

def format_large_number(num):
    """Format large numbers for display (e.g., 1,234,567 -> 1.23M)"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    if num >= 1e3:
        return f"{num/1e3:.2f}K"
    return f"{num:.2f}"

def format_percentage(pct):
    """Format percentage for display with colors"""
    color = 'green' if pct >= 0 else 'red'
    return f"<span style='color:{color}'>{pct:.2f}%</span>"

def create_candlestick_chart(data, title="Stock Price"):
    """Create a candlestick chart from OHLC data"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode='x unified'
    )
    
    return fig

def calculate_returns(prices, period=1):
    """Calculate returns over a specified period"""
    return prices.pct_change(period) * 100

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate the Sharpe ratio"""
    # Convert annual risk-free rate to match return frequency (assuming daily returns)
    rf_daily = (1 + risk_free_rate) ** (1/252) - 1
    
    excess_returns = returns - rf_daily
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)  # annualized

def calculate_drawdown(prices):
    """Calculate maximum drawdown"""
    # Calculate the cumulative returns
    cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
    
    # Calculate the running maximum
    running_max = cumulative.cummax()
    
    # Calculate the drawdown
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()  # Maximum drawdown

def calculate_beta(stock_returns, market_returns):
    """Calculate beta relative to the market"""
    # Calculate covariance
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    
    # Calculate market variance
    market_variance = np.var(market_returns)
    
    # Calculate beta
    return covariance / market_variance

def generate_performance_metrics(stock_data, benchmark_symbol='^GSPC'):
    """Generate comprehensive performance metrics for a stock"""
    # Calculate daily returns
    stock_returns = stock_data['Close'].pct_change().dropna()
    
    # Get benchmark data (S&P 500)
    benchmark = yf.download(benchmark_symbol, 
                            start=stock_data.index[0],
                            end=stock_data.index[-1],
                            progress=False)
    benchmark_returns = benchmark['Close'].pct_change().dropna()
    
    # Align the data
    common_dates = stock_returns.index.intersection(benchmark_returns.index)
    stock_returns = stock_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    
    # Calculate metrics
    metrics = {
        'Total Return (%)': ((stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1) * 100,
        'Annualized Return (%)': stock_returns.mean() * 252 * 100,  # 252 trading days in a year
        'Volatility (%)': stock_returns.std() * np.sqrt(252) * 100,  # Annualized
        'Sharpe Ratio': calculate_sharpe_ratio(stock_returns),
        'Max Drawdown (%)': calculate_drawdown(stock_data['Close']) * 100,
        'Beta': calculate_beta(stock_returns, benchmark_returns) if not benchmark_returns.empty else None,
        'Alpha (%)': (stock_returns.mean() - benchmark_returns.mean() * calculate_beta(stock_returns, benchmark_returns)) * 252 * 100 if not benchmark_returns.empty else None
    }
    
    return metrics
