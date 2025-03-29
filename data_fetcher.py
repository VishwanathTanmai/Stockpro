import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import random  # Just for generating sample data when real data is not available

# List of major cryptocurrencies with their Yahoo Finance tickers
MAJOR_CRYPTOCURRENCIES = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Binance Coin": "BNB-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "Cardano": "ADA-USD",
    "Avalanche": "AVAX-USD",
    "Dogecoin": "DOGE-USD",
    "Polkadot": "DOT-USD",
    "Polygon": "MATIC-USD",
    "Shiba Inu": "SHIB-USD",
    "Litecoin": "LTC-USD",
    "Uniswap": "UNI-USD",
    "Chainlink": "LINK-USD",
    "Bitcoin Cash": "BCH-USD",
    "Stellar": "XLM-USD",
    "Cosmos": "ATOM-USD",
    "Monero": "XMR-USD",
    "Tron": "TRX-USD",
    "Tezos": "XTZ-USD"
}

# Cache the stock data to avoid repeated API calls
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_data(symbol, period="1mo"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            return None
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None
        
def get_historical_stock_data(symbol, period="max"):
    """Fetch long-term historical stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            return None
        
        return data
    except Exception as e:
        st.error(f"Error fetching historical data for {symbol}: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_company_info(symbol):
    """Fetch company info from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Error fetching company info for {symbol}: {e}")
        return {}

def get_current_price(symbol):
    """Get the current price of a stock"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        
        if data.empty:
            return None
        
        return data['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching current price for {symbol}: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_market_indices(indices):
    """Fetch data for major market indices"""
    try:
        data = []
        for idx in indices:
            ticker = yf.Ticker(idx)
            hist = ticker.history(period="2d")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                
                data.append({
                    'Symbol': idx,
                    'Price': current,
                    'Change %': change
                })
        
        return data
    except Exception as e:
        st.error(f"Error fetching market indices: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_market_sentiment():
    """Get overall market sentiment data"""
    try:
        # This is a placeholder - in a real app, you would get this data from an API
        # For now, we'll simulate some data
        
        # Get S&P 500 data as a proxy for market sentiment
        sp500 = yf.Ticker("^GSPC")
        data = sp500.history(period="30d")
        
        # Calculate metrics based on S&P 500 data
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-5]  # 5 days ago
        change = ((current_price - prev_price) / prev_price) * 100
        
        # Calculate average volume
        avg_volume = data['Volume'].mean()
        current_volume = data['Volume'].iloc[-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        
        # Calculate volatility (standard deviation of returns)
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * 100
        volatility_30d = returns.rolling(window=30).std().iloc[-1] * 100
        volatility_change = volatility - volatility_30d
        
        # Fear & Greed index (simulated)
        # In a real app, you would get this from an API like CNN's Fear & Greed Index
        fear_greed_value = 50 + change  # Simple proxy
        fear_greed_value = max(min(fear_greed_value, 100), 0)  # Clamp between 0 and 100
        
        return {
            'fear_greed': fear_greed_value,
            'fear_greed_change': change,
            'volume': current_volume,
            'volume_change': volume_change,
            'volatility': volatility,
            'volatility_change': volatility_change
        }
    except Exception as e:
        st.error(f"Error fetching market sentiment: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_most_active_stocks():
    """Get most active stocks for the day"""
    try:
        # In a real app, you would get this from Yahoo Finance API
        # For now, let's use a predefined list of stocks and get their current data
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD"]
        
        data = []
        for symbol in popular_stocks:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="2d")
            
            if not hist.empty and 'shortName' in info:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                
                data.append({
                    'Symbol': symbol,
                    'Name': info['shortName'],
                    'Price': current,
                    'Change %': change,
                    'Volume': hist['Volume'].iloc[-1]
                })
        
        # Sort by volume (most active)
        if data:
            data.sort(key=lambda x: x['Volume'], reverse=True)
        
        return data
    except Exception as e:
        st.error(f"Error fetching most active stocks: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_market_news():
    """Get latest market news"""
    try:
        # In a real app, you would get this from a news API
        # For now, let's generate some synthetic news based on real stock data
        
        # Get S&P 500 data
        sp500 = yf.Ticker("^GSPC")
        sp_data = sp500.history(period="5d")
        last_change = ((sp_data['Close'].iloc[-1] - sp_data['Close'].iloc[-2]) / sp_data['Close'].iloc[-2]) * 100
        
        # Generate news based on market movement
        news_items = []
        
        # Current date
        now = datetime.now()
        
        if last_change > 0:
            news_items.append({
                'title': f"Markets Rally: S&P 500 Gains {abs(last_change):.2f}% as Investor Confidence Grows",
                'source': 'Financial Times',
                'published': (now - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
                'summary': "Stock markets surged today as positive economic data boosted investor confidence. Analysts point to strong employment figures and easing inflation concerns as catalysts for the rally.",
                'link': 'https://www.ft.com'
            })
        else:
            news_items.append({
                'title': f"Markets Decline: S&P 500 Drops {abs(last_change):.2f}% Amid Economic Concerns",
                'source': 'Wall Street Journal',
                'published': (now - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M'),
                'summary': "Stocks retreated today as investors digested disappointing corporate earnings and concerns about interest rates. Market volatility increased as traders reassessed growth prospects.",
                'link': 'https://www.wsj.com'
            })
        
        # Add some generic financial news
        news_items.extend([
            {
                'title': "Federal Reserve Signals Potential Rate Change in Upcoming Meeting",
                'source': 'CNBC',
                'published': (now - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M'),
                'summary': "Federal Reserve officials hinted at a possible adjustment to interest rates in their upcoming policy meeting. Market participants are closely monitoring economic indicators for clues about the central bank's next move.",
                'link': 'https://www.cnbc.com'
            },
            {
                'title': "Tech Stocks Lead Market as AI Innovations Drive Growth",
                'source': 'Bloomberg',
                'published': (now - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
                'summary': "Technology companies are outperforming the broader market as artificial intelligence advancements create new revenue opportunities. Analysts predict continued growth in the sector as businesses increase tech investments.",
                'link': 'https://www.bloomberg.com'
            },
            {
                'title': "Oil Prices Fluctuate Following OPEC+ Production Decision",
                'source': 'Reuters',
                'published': (now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M'),
                'summary': "Crude oil prices experienced volatility after OPEC+ announced its latest production targets. Energy analysts are divided on the long-term impact, with supply constraints competing against concerns about global demand.",
                'link': 'https://www.reuters.com'
            },
            {
                'title': "Retail Sales Data Surpasses Expectations, Boosting Consumer Sectors",
                'source': 'MarketWatch',
                'published': (now - timedelta(days=1, hours=4)).strftime('%Y-%m-%d %H:%M'),
                'summary': "The latest retail sales figures exceeded analyst projections, signaling robust consumer spending despite inflation pressures. Consumer discretionary stocks rallied on the news, with several retailers hitting new 52-week highs.",
                'link': 'https://www.marketwatch.com'
            }
        ])
        
        return news_items
    except Exception as e:
        st.error(f"Error fetching market news: {e}")
        return None

# Cryptocurrency-specific functions
@st.cache_data(ttl=300)  # Cache for 5 minutes (shorter time due to crypto volatility)
def get_crypto_data(symbol, period="1mo"):
    """Fetch cryptocurrency data from Yahoo Finance"""
    try:
        # For cryptocurrencies, we append -USD if it's not already there
        if not (symbol.endswith('-USD') or symbol.endswith('-USDT')):
            actual_symbol = f"{symbol}-USD"
        else:
            actual_symbol = symbol
            
        crypto = yf.Ticker(actual_symbol)
        data = crypto.history(period=period)
        
        if data.empty:
            return None
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for cryptocurrency {symbol}: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_top_cryptocurrencies(limit=20):
    """Get data for top cryptocurrencies"""
    try:
        data = []
        
        # Use our predefined list of major cryptocurrencies
        for name, symbol in list(MAJOR_CRYPTOCURRENCIES.items())[:limit]:
            crypto = yf.Ticker(symbol)
            hist = crypto.history(period="2d")
            info = crypto.info
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                if len(hist) > 1:
                    prev = hist['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                else:
                    change = 0
                
                market_cap = info.get('marketCap', None)
                
                data.append({
                    'Name': name,
                    'Symbol': symbol,
                    'Price': current,
                    'Change %': change,
                    'Market Cap': market_cap,
                    'Volume (24h)': hist['Volume'].iloc[-1] if 'Volume' in hist else None
                })
        
        # Sort by market cap if available
        if data:
            data.sort(key=lambda x: x['Market Cap'] if x['Market Cap'] is not None else 0, reverse=True)
        
        return data
    except Exception as e:
        st.error(f"Error fetching top cryptocurrencies: {e}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_crypto_info(symbol):
    """Fetch cryptocurrency info from Yahoo Finance"""
    try:
        # For cryptocurrencies, we append -USD if it's not already there
        if not (symbol.endswith('-USD') or symbol.endswith('-USDT')):
            actual_symbol = f"{symbol}-USD"
        else:
            actual_symbol = symbol
            
        crypto = yf.Ticker(actual_symbol)
        info = crypto.info
        
        # Add a specific field to identify it as a cryptocurrency
        info['isCryptocurrency'] = True
        
        return info
    except Exception as e:
        st.error(f"Error fetching cryptocurrency info for {symbol}: {e}")
        return {}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_crypto_market_overview():
    """Get overall cryptocurrency market metrics"""
    try:
        # Get Bitcoin data as a proxy for crypto market
        btc = yf.Ticker("BTC-USD")
        eth = yf.Ticker("ETH-USD")
        xrp = yf.Ticker("XRP-USD")
        ada = yf.Ticker("ADA-USD")
        sol = yf.Ticker("SOL-USD")
        
        btc_data = btc.history(period="30d")
        eth_data = eth.history(period="30d")
        
        # Calculate total market metrics using Bitcoin and Ethereum as proxies
        btc_market_cap = btc.info.get('marketCap', 0)
        eth_market_cap = eth.info.get('marketCap', 0)
        xrp_market_cap = xrp.info.get('marketCap', 0)
        ada_market_cap = ada.info.get('marketCap', 0)
        sol_market_cap = sol.info.get('marketCap', 0)
        
        # Total market cap estimation (these top coins represent a significant portion of the market)
        total_market_cap = btc_market_cap + eth_market_cap + xrp_market_cap + ada_market_cap + sol_market_cap
        
        # Get 24h trading volume data
        btc_volume = btc_data['Volume'].iloc[-1] * btc_data['Close'].iloc[-1]
        eth_volume = eth_data['Volume'].iloc[-1] * eth_data['Close'].iloc[-1]
        total_volume_24h = btc_volume + eth_volume  # Simplified estimation
        
        # Calculate metrics
        btc_current = btc_data['Close'].iloc[-1]
        btc_prev = btc_data['Close'].iloc[-2]
        btc_change = ((btc_current - btc_prev) / btc_prev) * 100
        
        eth_current = eth_data['Close'].iloc[-1]
        eth_prev = eth_data['Close'].iloc[-2]
        eth_change = ((eth_current - eth_prev) / eth_prev) * 100
        
        # Calculate volatility (standard deviation of returns)
        btc_returns = btc_data['Close'].pct_change().dropna()
        btc_volatility = btc_returns.std() * 100
        
        eth_returns = eth_data['Close'].pct_change().dropna()
        eth_volatility = eth_returns.std() * 100
        
        # Estimate market sentiment based on price changes
        market_change_24h = (btc_change + eth_change) / 2  # Simple proxy for overall market movement
        sentiment_value = 50 + market_change_24h  # Simple proxy
        sentiment_value = max(min(sentiment_value, 100), 0)  # Clamp between 0 and 100
        
        # BTC dominance calculation
        btc_dominance = (btc_market_cap / total_market_cap) * 100 if total_market_cap > 0 else 0
        
        # Additional metrics for enhanced market overview
        return {
            'total_market_cap': total_market_cap,
            'total_volume_24h': total_volume_24h,
            'btc_dominance': btc_dominance,
            'market_change_24h': market_change_24h,
            'sentiment': sentiment_value,
            'sentiment_change': market_change_24h,
            'btc_price': btc_current,
            'btc_change': btc_change,
            'eth_price': eth_current,
            'eth_change': eth_change,
            'volatility': (btc_volatility + eth_volatility) / 2,
            'active_cryptocurrencies': 10000,  # Approximate number
            'active_exchanges': 400,          # Approximate number
            'btc_volume_24h': btc_volume,
            'eth_volume_24h': eth_volume
        }
    except Exception as e:
        st.error(f"Error fetching crypto market overview: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_crypto_news():
    """Get latest cryptocurrency news"""
    try:
        # Get Bitcoin data to estimate market sentiment for news topics
        btc = yf.Ticker("BTC-USD")
        btc_data = btc.history(period="5d")
        last_change = ((btc_data['Close'].iloc[-1] - btc_data['Close'].iloc[-2]) / btc_data['Close'].iloc[-2]) * 100
        
        # Generate news based on market movement
        news_items = []
        
        # Current date
        now = datetime.now()
        
        if last_change > 0:
            news_items.append({
                'title': f"Bitcoin Surges {abs(last_change):.2f}%, Crypto Market Follows",
                'source': 'CoinDesk',
                'published': (now - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M'),
                'summary': "Bitcoin's rally brought positive momentum to the entire cryptocurrency market, with most major coins recording gains. Analysts point to institutional adoption and regulatory clarity as potential drivers.",
                'link': 'https://www.coindesk.com'
            })
        else:
            news_items.append({
                'title': f"Bitcoin Drops {abs(last_change):.2f}% Amid Market Uncertainty",
                'source': 'CryptoNews',
                'published': (now - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M'),
                'summary': "Bitcoin and other cryptocurrencies saw significant declines as market sentiment turned bearish. Traders point to regulatory concerns and macroeconomic factors affecting risk assets across the board.",
                'link': 'https://cryptonews.com'
            })
        
        # Add generic crypto news
        news_items.extend([
            {
                'title': "Ethereum Upgrade Anticipation Drives Price Action",
                'source': 'Cointelegraph',
                'published': (now - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'),
                'summary': "The upcoming Ethereum network upgrade has created anticipation among investors, driving increased trading volume. Developers promise improvements in scalability and reduced transaction fees.",
                'link': 'https://cointelegraph.com'
            },
            {
                'title': "New Institutional Cryptocurrency Fund Launches with $500M",
                'source': 'The Block',
                'published': (now - timedelta(days=1)).strftime('%Y-%m-%d %H:%M'),
                'summary': "A major asset management firm has announced a new cryptocurrency-focused fund with $500 million in initial investments. The fund will focus on Bitcoin, Ethereum, and other large-cap digital assets.",
                'link': 'https://www.theblock.co'
            },
            {
                'title': "Central Bank Digital Currencies Face Both Support and Criticism",
                'source': 'Decrypt',
                'published': (now - timedelta(days=2)).strftime('%Y-%m-%d %H:%M'),
                'summary': "As more countries explore Central Bank Digital Currencies (CBDCs), the cryptocurrency community remains divided. Proponents highlight efficiency gains, while critics worry about privacy implications and centralization.",
                'link': 'https://decrypt.co'
            }
        ])
        
        return news_items
    except Exception as e:
        st.error(f"Error fetching crypto news: {e}")
        return None
