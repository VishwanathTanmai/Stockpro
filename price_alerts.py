import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import data_fetcher
import database

def show_alerts_page():
    """Display the price alerts management page"""
    if not st.session_state.logged_in:
        st.warning("Please login to access the price alerts feature.")
        return
    
    st.header("Price Alerts")
    
    # Initialize alerts in session state if not present
    if 'alerts' not in st.session_state:
        st.session_state.alerts = load_user_alerts(st.session_state.username)
    
    # Tab selection
    tab1, tab2 = st.tabs(["Create Alert", "Manage Alerts"])
    
    with tab1:
        create_alert_interface()
    
    with tab2:
        manage_alerts_interface()

def create_alert_interface():
    """Interface for creating new price alerts"""
    st.subheader("Create New Alert")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.text_input("Enter stock symbol (e.g. AAPL, MSFT, GOOGL)", key="alert_symbol_input")
    
    # If a symbol is entered, show the alert creation form
    if symbol:
        symbol = symbol.upper().strip()
        
        # Get current price
        current_price = data_fetcher.get_current_price(symbol)
        
        if current_price is None:
            st.error(f"Could not find price data for symbol: {symbol}")
            return
        
        # Get company info
        company_info = data_fetcher.get_company_info(symbol)
        company_name = company_info.get('longName', symbol) if company_info else symbol
        
        # Display current info
        st.info(f"**{company_name} ({symbol})** - Current Price: **${current_price:.2f}**")
        
        # Alert parameters
        st.subheader("Alert Parameters")
        
        # Select alert type
        alert_type = st.radio(
            "Alert Type:",
            ["Price Above", "Price Below", "Price Change %", "Moving Average Crossover"],
            horizontal=True
        )
        
        if alert_type == "Price Above":
            trigger_value = st.number_input(
                "Trigger when price goes above ($):",
                min_value=0.01,
                max_value=float(current_price) * 2,
                value=float(current_price) * 1.05,  # Default to 5% above current price
                step=0.01
            )
            
            alert_message = f"Price of {symbol} ({company_name}) above ${trigger_value:.2f}"
        
        elif alert_type == "Price Below":
            trigger_value = st.number_input(
                "Trigger when price goes below ($):",
                min_value=0.01,
                max_value=float(current_price) * 2,
                value=float(current_price) * 0.95,  # Default to 5% below current price
                step=0.01
            )
            
            alert_message = f"Price of {symbol} ({company_name}) below ${trigger_value:.2f}"
        
        elif alert_type == "Price Change %":
            change_direction = st.radio(
                "Direction:",
                ["Increases by", "Decreases by"],
                horizontal=True
            )
            
            percentage = st.number_input(
                "Percentage (%):",
                min_value=0.1,
                max_value=100.0,
                value=5.0,
                step=0.1
            )
            
            # Calculate trigger value
            if change_direction == "Increases by":
                trigger_value = current_price * (1 + percentage/100)
                alert_message = f"Price of {symbol} ({company_name}) increases by {percentage}% from ${current_price:.2f}"
            else:
                trigger_value = current_price * (1 - percentage/100)
                alert_message = f"Price of {symbol} ({company_name}) decreases by {percentage}% from ${current_price:.2f}"
        
        else:  # Moving Average Crossover
            ma_type = st.radio(
                "Crossover Type:",
                ["Price crosses above MA", "Price crosses below MA", "MA crosses another MA"],
                horizontal=True
            )
            
            if ma_type == "MA crosses another MA":
                ma1 = st.selectbox(
                    "First Moving Average:",
                    ["5-day", "10-day", "20-day", "50-day", "100-day", "200-day"],
                    index=2  # Default to 20-day
                )
                
                ma2 = st.selectbox(
                    "Second Moving Average:",
                    ["5-day", "10-day", "20-day", "50-day", "100-day", "200-day"],
                    index=3  # Default to 50-day
                )
                
                cross_direction = st.radio(
                    "Direction:",
                    [f"{ma1} crosses above {ma2}", f"{ma1} crosses below {ma2}"],
                    horizontal=True
                )
                
                trigger_value = 0  # Placeholder, actual calculation is more complex
                alert_message = f"{cross_direction} for {symbol} ({company_name})"
            
            else:
                ma_period = st.selectbox(
                    "Moving Average Period:",
                    ["5-day", "10-day", "20-day", "50-day", "100-day", "200-day"],
                    index=3  # Default to 50-day
                )
                
                trigger_value = 0  # Placeholder, actual calculation is more complex
                alert_message = f"Price of {symbol} ({company_name}) {ma_type} {ma_period}"
        
        # Alert expiration
        st.subheader("Alert Expiration")
        
        expiration_type = st.radio(
            "Expiration:",
            ["1 day", "1 week", "1 month", "Never (until triggered)"],
            horizontal=True,
            index=2  # Default to 1 month
        )
        
        if expiration_type == "Never (until triggered)":
            expiration_date = None
        else:
            days_map = {"1 day": 1, "1 week": 7, "1 month": 30}
            expiration_date = datetime.now() + timedelta(days=days_map[expiration_type])
        
        # Notification settings
        st.subheader("Notification Settings")
        
        notify_once = st.checkbox("Notify only once (disable after triggering)", value=True)
        
        # Create alert button
        if st.button("Create Alert"):
            alert = {
                "id": f"{datetime.now().timestamp()}",
                "symbol": symbol,
                "company_name": company_name,
                "current_price": current_price,
                "alert_type": alert_type,
                "trigger_value": trigger_value,
                "message": alert_message,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "expiration_date": expiration_date.strftime("%Y-%m-%d %H:%M:%S") if expiration_date else None,
                "notify_once": notify_once,
                "status": "Active",
                "triggered": False,
                "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add specific parameters based on alert type
            if alert_type == "Moving Average Crossover":
                if ma_type == "MA crosses another MA":
                    alert["ma1"] = ma1
                    alert["ma2"] = ma2
                    alert["cross_direction"] = cross_direction
                else:
                    alert["ma_period"] = ma_period
                    alert["cross_direction"] = ma_type
            elif alert_type == "Price Change %":
                alert["change_direction"] = change_direction
                alert["percentage"] = percentage
            
            # Save alert to session state
            if 'alerts' not in st.session_state:
                st.session_state.alerts = []
            
            st.session_state.alerts.append(alert)
            
            # Save to database
            save_user_alerts(st.session_state.username, st.session_state.alerts)
            
            st.success(f"Alert created: {alert_message}")
            
            # Show visualization of the alert
            show_alert_visualization(symbol, current_price, alert_type, trigger_value)
    
    else:
        st.info("Enter a stock symbol to create an alert.")

def manage_alerts_interface():
    """Interface for managing existing alerts"""
    st.subheader("Your Alerts")
    
    if 'alerts' not in st.session_state or not st.session_state.alerts:
        st.info("You don't have any alerts set up yet.")
        return
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        status_filter = st.selectbox(
            "Filter by status:",
            ["All", "Active", "Triggered", "Expired"],
            index=0
        )
    
    with filter_col2:
        if status_filter == "All":
            alerts_to_show = st.session_state.alerts
        else:
            alerts_to_show = [alert for alert in st.session_state.alerts 
                             if alert["status"] == status_filter]
    
    # Check for alerts that meet the filter criteria
    if not alerts_to_show:
        st.info(f"No {status_filter.lower()} alerts found.")
        return
    
    # Display alerts
    for i, alert in enumerate(alerts_to_show):
        with st.expander(f"{alert['symbol']} - {alert['message']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Symbol:** {alert['symbol']} ({alert['company_name']})")
                st.write(f"**Alert Type:** {alert['alert_type']}")
                st.write(f"**Created:** {alert['created_date']}")
                
                if alert['expiration_date']:
                    expiration_date = datetime.strptime(alert['expiration_date'], "%Y-%m-%d %H:%M:%S")
                    if expiration_date < datetime.now():
                        status = "Expired"
                    else:
                        status = alert['status']
                        st.write(f"**Expires:** {alert['expiration_date']}")
                else:
                    status = alert['status']
                    st.write("**Expires:** Never (until triggered)")
                
                st.write(f"**Status:** {status}")
                
                if alert['triggered']:
                    st.write("**Triggered:** Yes")
                
                # Alert details
                if alert['alert_type'] == "Price Above":
                    st.write(f"**Trigger Price:** ${alert['trigger_value']:.2f}")
                    st.write(f"**Price at Creation:** ${alert['current_price']:.2f}")
                
                elif alert['alert_type'] == "Price Below":
                    st.write(f"**Trigger Price:** ${alert['trigger_value']:.2f}")
                    st.write(f"**Price at Creation:** ${alert['current_price']:.2f}")
                
                elif alert['alert_type'] == "Price Change %":
                    st.write(f"**Change Direction:** {alert.get('change_direction', 'N/A')}")
                    st.write(f"**Percentage:** {alert.get('percentage', 0):.1f}%")
                    st.write(f"**Price at Creation:** ${alert['current_price']:.2f}")
                    st.write(f"**Trigger Price:** ${alert['trigger_value']:.2f}")
                
                elif alert['alert_type'] == "Moving Average Crossover":
                    if "ma1" in alert and "ma2" in alert:
                        st.write(f"**Crossover Type:** {alert.get('ma1', 'N/A')} crosses {alert.get('ma2', 'N/A')}")
                        st.write(f"**Direction:** {alert.get('cross_direction', 'N/A')}")
                    else:
                        st.write(f"**MA Period:** {alert.get('ma_period', 'N/A')}")
                        st.write(f"**Cross Direction:** {alert.get('cross_direction', 'N/A')}")
            
            with col2:
                # Current price and status check
                current_price = data_fetcher.get_current_price(alert['symbol'])
                
                if current_price is not None:
                    price_change = ((current_price / alert['current_price']) - 1) * 100
                    price_color = "green" if price_change >= 0 else "red"
                    
                    st.markdown(f"""
                    **Current Price:** ${current_price:.2f}
                    
                    <span style='color:{price_color}'>Change: {price_change:.2f}%</span>
                    """, unsafe_allow_html=True)
                    
                    # Check if alert should be triggered
                    alert_triggered = check_alert_condition(alert, current_price)
                    
                    if alert_triggered and not alert['triggered']:
                        st.success("Alert condition is met!")
                        
                        # Update alert status
                        alert['triggered'] = True
                        alert['status'] = "Triggered"
                        alert['trigger_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Save updated alerts
                        save_user_alerts(st.session_state.username, st.session_state.alerts)
                
                # Alert actions
                if st.button("Delete Alert", key=f"delete_{i}"):
                    # Remove alert from session state
                    st.session_state.alerts.remove(alert)
                    
                    # Save updated alerts
                    save_user_alerts(st.session_state.username, st.session_state.alerts)
                    
                    st.success("Alert deleted successfully.")
                    st.rerun()
                
                if alert['status'] == "Active":
                    if st.button("Pause Alert", key=f"pause_{i}"):
                        # Update alert status
                        alert['status'] = "Paused"
                        
                        # Save updated alerts
                        save_user_alerts(st.session_state.username, st.session_state.alerts)
                        
                        st.success("Alert paused successfully.")
                        st.rerun()
                
                elif alert['status'] == "Paused":
                    if st.button("Activate Alert", key=f"activate_{i}"):
                        # Update alert status
                        alert['status'] = "Active"
                        
                        # Save updated alerts
                        save_user_alerts(st.session_state.username, st.session_state.alerts)
                        
                        st.success("Alert activated successfully.")
                        st.rerun()
    
    # Clear all alerts button
    if st.button("Clear All Alerts"):
        # Confirm before clearing
        if st.checkbox("I confirm I want to delete all alerts", value=False):
            # Clear alerts
            st.session_state.alerts = []
            
            # Save empty alerts
            save_user_alerts(st.session_state.username, [])
            
            st.success("All alerts cleared successfully.")
            st.rerun()

def check_alert_condition(alert, current_price):
    """Check if an alert condition is met based on current price"""
    if alert['status'] != "Active":
        return False
    
    alert_type = alert['alert_type']
    
    if alert_type == "Price Above":
        return current_price > alert['trigger_value']
    
    elif alert_type == "Price Below":
        return current_price < alert['trigger_value']
    
    elif alert_type == "Price Change %":
        change_direction = alert.get('change_direction')
        
        if change_direction == "Increases by":
            return current_price >= alert['trigger_value']
        else:  # Decreases by
            return current_price <= alert['trigger_value']
    
    # For Moving Average Crossover, we'd need historical data to check
    # This is simplified for demo purposes
    return False

def show_alert_visualization(symbol, current_price, alert_type, trigger_value):
    """Create a visualization of the alert"""
    # Get historical data for context
    historical_data = data_fetcher.get_stock_data(symbol, period="1mo")
    
    if historical_data is None or historical_data.empty:
        return
    
    # Create a figure
    fig = go.Figure()
    
    # Add price history
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add alert line
    if alert_type in ["Price Above", "Price Below", "Price Change %"]:
        fig.add_trace(go.Scatter(
            x=[historical_data.index[0], historical_data.index[-1]],
            y=[trigger_value, trigger_value],
            mode='lines',
            name='Alert Level',
            line=dict(color='red', width=2, dash='dash')
        ))
    
    # Add annotation for current price
    fig.add_annotation(
        x=historical_data.index[-1],
        y=current_price,
        text=f"Current: ${current_price:.2f}",
        showarrow=True,
        arrowhead=1,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
    
    # Add annotation for alert level
    if alert_type in ["Price Above", "Price Below", "Price Change %"]:
        fig.add_annotation(
            x=historical_data.index[0],
            y=trigger_value,
            text=f"Alert: ${trigger_value:.2f}",
            showarrow=True,
            arrowhead=1,
            bgcolor="rgba(255, 200, 200, 0.8)"
        )
    
    # Set layout
    fig.update_layout(
        height=400,
        title=f"Alert Visualization for {symbol}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def load_user_alerts(username):
    """Load alerts from database for a user"""
    # Get user data
    user_data = database.get_user_data(username)
    
    # Check if alerts exist
    if user_data and 'alerts' in user_data:
        return user_data['alerts']
    
    return []

def save_user_alerts(username, alerts):
    """Save alerts to database for a user"""
    # Get user data
    user_data = database.get_user_data(username)
    
    if user_data:
        # Update alerts
        user_data['alerts'] = alerts
        
        # Save updated user data
        database.update_user_data(username, user_data)
    
    return True