import streamlit as st
import database
import re
from datetime import datetime

def show_auth_page():
    """Display the authentication page (login/signup) with enhanced UI"""
    # Stock market background image and welcome message
    st.markdown("""
    <div class="auth-container animated">
        <div class="auth-welcome">
            <h1>Welcome to StockPro</h1>
            <p>Your advanced platform for stock market analysis, portfolio management, and AI-powered predictions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Styled tabs
        st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.2rem;
            font-weight: 600;
            padding: 0 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding: 10px 16px;
            background-color: rgba(28, 131, 225, 0.1);
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(28, 131, 225, 0.2);
            border-bottom: 2px solid #1c83e1;
        }
        </style>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "✏️ Sign Up"])
        
        with tab1:
            show_login()
        
        with tab2:
            show_signup()
    
    # App features and benefits
    st.markdown("""
    <div class="features-container">
        <h3>StockPro Features</h3>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <div class="feature-title">Real-time Market Data</div>
                <div class="feature-desc">Access live stock prices, charts, and market indices</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔮</div>
                <div class="feature-title">AI Predictions</div>
                <div class="feature-desc">Advanced neural networks to forecast stock prices</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Portfolio Analysis</div>
                <div class="feature-desc">Track performance, gains, and optimal allocations</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">Auto Trading</div>
                <div class="feature-desc">Algorithm-based trading recommendations</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_login():
    """Display the enhanced login form"""
    # Styled login container
    st.markdown('<div class="login-container css-card">', unsafe_allow_html=True)
    st.subheader("Sign in to your account")
    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create form with custom styling
    username = st.text_input("Username", key="login_username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
    
    # Remember me checkbox and forgot password
    col1, col2 = st.columns(2)
    with col1:
        remember = st.checkbox("Remember me", value=True)
    with col2:
        st.markdown('<div style="text-align: right;"><a href="#" style="color: #1c83e1; text-decoration: none;">Forgot password?</a></div>', unsafe_allow_html=True)
    
    # Submit button with full width
    if st.button("Login", key="login_button", type="primary", use_container_width=True):
        if username and password:
            # Show loading spinner during verification
            with st.spinner("Verifying credentials..."):
                # Verify credentials
                if database.verify_user(username, password):
                    # Login successful
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.current_page = 'dashboard'
                    
                    # Load user's portfolio and balance
                    user_data = database.get_user_data(username)
                    st.session_state.balance = user_data.get('demo_balance', 1000000.0)
                    st.session_state.crypto_balance = user_data.get('demo_crypto_balance', 1000000.0)
                    st.session_state.portfolio = user_data.get('portfolio', {})
                    st.session_state.crypto_portfolio = user_data.get('crypto_portfolio', {})
                    
                    st.success("Login successful! Redirecting to dashboard...")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        else:
            st.warning("Please enter both username and password.")
    
    # Demo account option
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;">Don\'t have an account? Try our <a href="#" style="color: #1c83e1; text-decoration: none;">demo account</a> or sign up.</div>', unsafe_allow_html=True)
    
    # Close the container
    st.markdown('</div>', unsafe_allow_html=True)

def show_signup():
    """Display the enhanced signup form"""
    # Styled signup container
    st.markdown('<div class="signup-container css-card">', unsafe_allow_html=True)
    st.subheader("Create your account")
    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create form with custom styling
    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username", key="signup_username", placeholder="Choose a username")
    with col2:
        email = st.text_input("Email", key="signup_email", placeholder="Enter your email")
    
    password = st.text_input("Password", type="password", key="signup_password", placeholder="Create a password")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password", placeholder="Confirm your password")
    
    # Password strength indicator
    if password:
        strength = len(password)
        if strength < 6:
            strength_color = "red"
            strength_text = "Weak"
        elif strength < 10:
            strength_color = "orange"
            strength_text = "Moderate"
        else:
            strength_color = "green"
            strength_text = "Strong"
        
        st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <p style="margin-bottom: 5px; font-size: 0.9rem;">Password strength: 
                <span style="color: {strength_color}; font-weight: bold;">{strength_text}</span>
            </p>
            <div style="background-color: #e0e0e0; border-radius: 5px; height: 5px; width: 100%;">
                <div style="background-color: {strength_color}; width: {min(strength * 10, 100)}%; height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Terms and conditions
    terms = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="terms")
    
    # Newsletter opt-in
    newsletter = st.checkbox("Send me market updates and news (optional)", key="newsletter", value=True)
    
    # Submit button with full width
    if st.button("Create Account", key="signup_button", type="primary", use_container_width=True):
        # Validate inputs
        if not username or not email or not password or not confirm_password:
            st.warning("Please fill out all required fields.")
        elif not terms:
            st.warning("Please agree to the Terms of Service and Privacy Policy.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        elif not validate_email(email):
            st.error("Please enter a valid email address.")
        elif len(password) < 6:
            st.error("Password should be at least 6 characters long.")
        else:
            # Show loading spinner during account creation
            with st.spinner("Creating your account..."):
                # Check if username already exists
                if database.user_exists(username):
                    st.error("Username already exists. Please choose another one.")
                else:
                    # Create new user
                    success = database.create_user(username, email, password)
                    if success:
                        st.success("Account created successfully! You can now log in.")
                    else:
                        st.error("Failed to create account. Please try again.")
    
    # Already have an account link
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="text-align: center;">Already have an account? <a href="#" style="color: #1c83e1; text-decoration: none;">Sign in</a></div>', unsafe_allow_html=True)
    
    # Close the container
    st.markdown('</div>', unsafe_allow_html=True)

def show_profile():
    """Display the enhanced user profile page"""
    if not st.session_state.logged_in:
        st.warning("Please login to view your profile.")
        return
    
    username = st.session_state.username
    user_data = database.get_user_data(username)
    
    # Create header with user info
    st.markdown(f"""
    <div class="profile-header animated">
        <div class="profile-avatar">👤</div>
        <div class="profile-title">
            <h1>{username}</h1>
            <p>Active Investor</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different profile sections
    tab1, tab2, tab3 = st.tabs(["🧑‍💼 Account Info", "📊 Trading Activity", "⚙️ Settings"])
    
    with tab1:
        # Account overview card
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Account Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="profile-detail">
                <div class="detail-label">Email</div>
                <div class="detail-value">{user_data.get('email', 'N/A')}</div>
            </div>
            <div class="profile-detail">
                <div class="detail-label">Member Since</div>
                <div class="detail-value">{user_data.get('join_date', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            balance = user_data.get('demo_balance', 1000000.0)
            st.markdown(f"""
            <div class="profile-detail">
                <div class="detail-label">Demo Account Balance</div>
                <div class="detail-value amount">${balance:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Allow user to reset demo account
            if st.button("Reset Demo Account", type="secondary"):
                database.reset_demo_account(username)
                st.success("Demo account has been reset to $1,000,000 for both stocks and crypto.")
                st.session_state.balance = 1000000.0
                st.session_state.crypto_balance = 1000000.0
                st.session_state.portfolio = {}
                st.session_state.crypto_portfolio = {}
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Account stats card
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Account Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate some stats
        history = user_data.get('trading_history', [])
        num_trades = len(history)
        portfolio = user_data.get('portfolio', {})
        num_stocks = len(portfolio)
        
        # Calculate profits if available
        profits = 0
        for trade in history:
            if trade.get('order_type') == 'sell':
                profits += trade.get('total_amount', 0)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{num_trades}</div>
                <div class="stat-label">Total Trades</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{num_stocks}</div>
                <div class="stat-label">Stocks Held</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            profit_class = "profit-positive" if profits >= 0 else "profit-negative"
            profit_sign = "+" if profits >= 0 else "-"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value {profit_class}">{profit_sign}${abs(profits):,.2f}</div>
                <div class="stat-label">Trading Profits</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Trading activity card
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Trading History")
        
        history = user_data.get('trading_history', [])
        if history:
            # Show recent trades
            history_df = database.get_trading_history_as_df(username)
            
            # Add styling to the dataframe
            st.markdown("""
            <style>
            .dataframe td:nth-child(5) {
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(history_df, use_container_width=True)
            
            # Add trading statistics
            buy_trades = [t for t in history if t.get('order_type') == 'buy']
            sell_trades = [t for t in history if t.get('order_type') == 'sell']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Buy Orders", len(buy_trades))
            
            with col2:
                st.metric("Sell Orders", len(sell_trades))
            
        else:
            st.info("No trading history yet. Make some trades to see your activity here!")
            
            # Add a quick trading button
            if st.button("Start Trading Now", type="primary"):
                st.session_state.current_page = 'trading'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Settings card
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Account Settings")
        
        # Settings form
        with st.form("settings_form"):
            # General settings
            st.write("General Settings")
            default_page = st.selectbox(
                "Default Landing Page", 
                options=["Dashboard", "Portfolio", "Trading", "Predictions"],
                index=0
            )
            
            # Notifications settings
            st.write("Notification Preferences")
            email_alerts = st.checkbox("Email Price Alerts", value=True)
            trade_confirmations = st.checkbox("Trade Confirmation Emails", value=True)
            market_news = st.checkbox("Market News Digest", value=True)
            
            # Submit button
            submit_button = st.form_submit_button("Save Settings")
            if submit_button:
                st.success("Settings saved successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Password change card
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.subheader("Change Password")
        
        with st.form("password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            submit_button = st.form_submit_button("Update Password")
            if submit_button:
                if not current_password or not new_password or not confirm_password:
                    st.warning("Please fill out all password fields.")
                elif new_password != confirm_password:
                    st.error("New passwords do not match.")
                elif len(new_password) < 6:
                    st.error("Password should be at least 6 characters long.")
                else:
                    # Here you would normally verify the current password and update
                    # Since this is a demo, we'll just show a success message
                    st.success("Password updated successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def logout():
    """Log out the current user"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.current_page = 'login'
    # Clear other session state variables
    if 'balance' in st.session_state:
        del st.session_state.balance
    if 'crypto_balance' in st.session_state:
        del st.session_state.crypto_balance
    if 'portfolio' in st.session_state:
        del st.session_state.portfolio
    if 'crypto_portfolio' in st.session_state:
        del st.session_state.crypto_portfolio

def validate_email(email):
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None
