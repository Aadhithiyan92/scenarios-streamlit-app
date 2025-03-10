import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Supply Chain Model", layout="wide")

st.title("Supply Chain Modeling: Traditional vs. Systems Dynamics")
st.write("""
This application demonstrates the fundamental difference between traditional forecasting 
and systems dynamics modeling in supply chain management.
""")

# Sidebar controls
st.sidebar.header("Model Parameters")
weeks_to_simulate = st.sidebar.slider("Weeks to Simulate", 20, 52, 40)
initial_customer_demand = st.sidebar.slider("Initial Customer Demand (units/week)", 100, 1000, 500)
demand_increase_week = st.sidebar.slider("Week of Demand Increase", 5, 15, 10)
demand_increase_percent = st.sidebar.slider("Demand Increase (%)", 10, 50, 20)

# Traditional Model Parameters
st.sidebar.header("Traditional Model Settings")
safety_stock_weeks = st.sidebar.slider("Safety Stock (weeks of demand)", 1.0, 4.0, 2.0, 0.5)
lead_time_weeks = st.sidebar.slider("Order Lead Time (weeks)", 1, 4, 2)

# Systems Dynamics Parameters
st.sidebar.header("Systems Dynamics Model Settings")
information_delay = st.sidebar.slider("Information Delay (weeks)", 0, 3, 1)
perception_weight = st.sidebar.slider("Weight on Recent Demand (%)", 10, 90, 30)
stockout_effect = st.sidebar.slider("Customer Response to Stockouts (0-1)", 0.0, 1.0, 0.5, 0.1)

# Create time periods
weeks = np.arange(1, weeks_to_simulate + 1)

# Generate actual customer demand pattern
def generate_customer_demand(initial_demand, increase_week, increase_pct, num_weeks):
    demand = np.zeros(num_weeks)
    # Add some randomness to base demand (±10%)
    noise = np.random.normal(0, 0.1, num_weeks)
    
    for week in range(num_weeks):
        if week < increase_week:
            base_demand = initial_demand
        else:
            base_demand = initial_demand * (1 + increase_pct/100)
        
        # Add noise and ensure demand is positive
        demand[week] = max(10, base_demand * (1 + noise[week]))
    
    return demand

# Customer demand is the same for both models
customer_demand = generate_customer_demand(
    initial_customer_demand, demand_increase_week, demand_increase_percent, weeks_to_simulate
)

# Function to simulate traditional forecasting model
def traditional_model(customer_demand, safety_stock_weeks, lead_time_weeks):
    num_weeks = len(customer_demand)
    
    # Initialize arrays
    inventory = np.zeros(num_weeks)
    orders = np.zeros(num_weeks)
    shipments_received = np.zeros(num_weeks)
    stockouts = np.zeros(num_weeks)
    
    # Initial conditions
    inventory[0] = safety_stock_weeks * customer_demand[0]
    
    # Simple order-up-to policy based on forecasted demand and safety stock
    for week in range(num_weeks):
        # Calculate forecast - simple moving average of last 4 weeks or fewer if at beginning
        if week < 4:
            forecast = np.mean(customer_demand[0:max(1, week)])
        else:
            forecast = np.mean(customer_demand[week-4:week])
        
        # Calculate target inventory (forecast * (lead time + safety stock))
        target_inventory = forecast * (lead_time_weeks + safety_stock_weeks)
        
        # Calculate order quantity
        if week < num_weeks - lead_time_weeks:
            pipeline_inventory = sum(orders[max(0, week-lead_time_weeks+1):week+1])
            order_quantity = max(0, target_inventory - inventory[week] - pipeline_inventory)
            orders[week] = order_quantity
        
        # Receive shipments after lead time
        if week >= lead_time_weeks:
            shipments_received[week] = orders[week - lead_time_weeks]
        
        # Update inventory
        if week > 0:
            inventory[week] = inventory[week-1] + shipments_received[week] - customer_demand[week-1]
            
            # Check for stockouts
            if inventory[week] < 0:
                stockouts[week] = abs(inventory[week])
                inventory[week] = 0
            else:
                stockouts[week] = 0
    
    return inventory, orders, shipments_received, stockouts

# Function to simulate systems dynamics model with feedback loops
def systems_dynamics_model(customer_demand, safety_stock_weeks, lead_time_weeks, info_delay, perception_weight, stockout_effect):
    num_weeks = len(customer_demand)
    
    # Initialize arrays
    inventory = np.zeros(num_weeks)
    perceived_demand = np.zeros(num_weeks)
    orders = np.zeros(num_weeks)
    shipments_received = np.zeros(num_weeks)
    stockouts = np.zeros(num_weeks)
    customer_satisfaction = np.ones(num_weeks)  # Start with full satisfaction
    supplier_capacity = np.zeros(num_weeks)
    supplier_capacity_utilization = np.zeros(num_weeks)
    
    # Initial conditions
    inventory[0] = safety_stock_weeks * customer_demand[0]
    perceived_demand[0] = customer_demand[0]
    supplier_capacity[0] = 1.5 * customer_demand[0]  # Initial capacity is 150% of demand
    
    # Systems dynamics simulation with feedback loops
    for week in range(1, num_weeks):
        # Update perceived demand with information delay and anchoring bias
        # (weight on new information vs. previous perception)
        if week > info_delay:
            new_info = customer_demand[week - info_delay] * customer_satisfaction[week-1]
            perceived_demand[week] = perceived_demand[week-1] * (1 - perception_weight/100) + \
                                    new_info * (perception_weight/100)
        else:
            perceived_demand[week] = perceived_demand[week-1]
        
        # Calculate target inventory (perceived demand * (lead time + safety stock))
        target_inventory = perceived_demand[week] * (lead_time_weeks + safety_stock_weeks)
        
        # Calculate order quantity with pipeline inventory consideration
        pipeline_inventory = sum(orders[max(0, week-lead_time_weeks):week])
        desired_order = max(0, target_inventory - inventory[week-1] - pipeline_inventory)
        
        # Supplier constraints - capacity grows or shrinks based on utilization
        supplier_capacity_utilization[week] = min(1.0, sum(orders[max(0, week-8):week]) / 
                                              (8 * supplier_capacity[week-1]))
        
        # Supplier capacity adjusts based on recent utilization
        if supplier_capacity_utilization[week] > 0.9:
            # Capacity increases when utilization is high
            supplier_capacity[week] = supplier_capacity[week-1] * 1.05
        elif supplier_capacity_utilization[week] < 0.7:
            # Capacity decreases when utilization is low
            supplier_capacity[week] = supplier_capacity[week-1] * 0.98
        else:
            supplier_capacity[week] = supplier_capacity[week-1]
        
        # Constrain orders by supplier capacity
        orders[week] = min(desired_order, supplier_capacity[week])
        
        # Receive shipments after lead time
        if week >= lead_time_weeks:
            shipments_received[week] = orders[week - lead_time_weeks]
        
        # Update inventory
        inventory[week] = inventory[week-1] + shipments_received[week] - customer_demand[week-1]
        
        # Check for stockouts and impact on customer satisfaction
        if inventory[week] < 0:
            stockouts[week] = abs(inventory[week])
            inventory[week] = 0
            # Customer satisfaction decreases with stockouts
            customer_satisfaction[week] = max(0.5, customer_satisfaction[week-1] * (1 - stockout_effect))
        else:
            stockouts[week] = 0
            # Customer satisfaction recovers slowly
            customer_satisfaction[week] = min(1.0, customer_satisfaction[week-1] + 0.05)
    
    return inventory, perceived_demand, orders, shipments_received, stockouts, customer_satisfaction, supplier_capacity

# Run the models
trad_inventory, trad_orders, trad_shipments, trad_stockouts = traditional_model(
    customer_demand, safety_stock_weeks, lead_time_weeks
)

sd_inventory, sd_perceived_demand, sd_orders, sd_shipments, sd_stockouts, sd_satisfaction, sd_capacity = systems_dynamics_model(
    customer_demand, safety_stock_weeks, lead_time_weeks, information_delay, perception_weight, stockout_effect
)

# Create DataFrames for plotting
traditional_df = pd.DataFrame({
    'Week': weeks,
    'Customer Demand': customer_demand,
    'Inventory': trad_inventory,
    'Orders Placed': trad_orders,
    'Shipments Received': trad_shipments,
    'Stockouts': trad_stockouts
})

sd_df = pd.DataFrame({
    'Week': weeks,
    'Customer Demand': customer_demand,
    'Perceived Demand': sd_perceived_demand,
    'Inventory': sd_inventory,
    'Orders Placed': sd_orders,
    'Shipments Received': sd_shipments,
    'Stockouts': sd_stockouts,
    'Customer Satisfaction': sd_satisfaction,
    'Supplier Capacity': sd_capacity
})

# Main content
st.header("Comparing Model Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Forecasting Model")
    st.write("""
    Traditional forecasting models typically use historical data to predict future demand
    and maintain inventory levels based on safety stock calculations. They often miss
    complex interactions and feedback loops in the supply chain.
    """)
    
    # Create plotly figure for traditional model
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(
        go.Scatter(x=weeks, y=customer_demand, name="Customer Demand", line=dict(color="blue", width=2)),
        secondary_y=False,
    )
    
    fig1.add_trace(
        go.Scatter(x=weeks, y=trad_inventory, name="Inventory", line=dict(color="green", width=3)),
        secondary_y=True,
    )
    
    fig1.add_trace(
        go.Scatter(x=weeks, y=trad_orders, name="Orders", line=dict(color="red", width=2, dash="dash")),
        secondary_y=False,
    )
    
    # Add a vertical line at demand increase week
    fig1.add_vline(x=demand_increase_week, line_width=2, line_dash="dash", line_color="gray")
    fig1.add_annotation(x=demand_increase_week, y=max(customer_demand)*1.1, 
                      text=f"Demand increases by {demand_increase_percent}%", showarrow=False)
    
    fig1.update_layout(
        title_text="Traditional Model: Inventory and Orders",
        xaxis_title="Week",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig1.update_yaxes(title_text="Units/Week", secondary_y=False)
    fig1.update_yaxes(title_text="Inventory Units", secondary_y=True)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Simple order-up-to policy based on forecasted demand
    - Fixed lead times and safety stock levels
    - No consideration of supplier constraints
    - Customer behavior doesn't change in response to stockouts
    - No adjustment for long-term trends
    """)

with col2:
    st.subheader("Systems Dynamics Model")
    st.write("""
    Systems dynamics captures feedback loops, delays, and non-linear relationships
    in how the supply chain responds to changing demand over time. It reveals the
    "bullwhip effect" and other emergent behaviors.
    """)
    
    # Create plotly figure for systems dynamics model
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(x=weeks, y=customer_demand, name="Customer Demand", line=dict(color="blue", width=2)),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(x=weeks, y=sd_perceived_demand, name="Perceived Demand", 
                  line=dict(color="purple", width=2, dash="dot")),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(x=weeks, y=sd_inventory, name="Inventory", line=dict(color="green", width=3)),
        secondary_y=True,
    )
    
    fig2.add_trace(
        go.Scatter(x=weeks, y=sd_orders, name="Orders", line=dict(color="red", width=2, dash="dash")),
        secondary_y=False,
    )
    
    # Add a vertical line at demand increase week
    fig2.add_vline(x=demand_increase_week, line_width=2, line_dash="dash", line_color="gray")
    fig2.add_annotation(x=demand_increase_week, y=max(customer_demand)*1.1, 
                      text=f"Demand increases by {demand_increase_percent}%", showarrow=False)
    
    fig2.update_layout(
        title_text="Systems Dynamics Model: Inventory and Orders",
        xaxis_title="Week",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig2.update_yaxes(title_text="Units/Week", secondary_y=False)
    fig2.update_yaxes(title_text="Inventory Units", secondary_y=True)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Feedback loops between inventory, orders, and customer behavior
    - Information delays affect perceived demand
    - Supplier capacity constraints create additional dynamics
    - Customer satisfaction changes based on product availability
    - Emergent behaviors like the "bullwhip effect" can be observed
    """)

# Systems Dynamics Key Components
st.header("Systems Dynamics: Revealing the Underlying Structure")

st.write("""
The power of systems dynamics lies in revealing how feedback loops create supply chain behaviors
that traditional models miss. Below are additional factors that influence the system behavior:
""")

# Create 3-panel chart to show key drivers in systems dynamics model
fig3 = make_subplots(rows=2, cols=1, 
                    subplot_titles=("Customer Satisfaction & Stockouts", "Supplier Capacity"))

fig3.add_trace(
    go.Scatter(x=weeks, y=sd_satisfaction, name="Customer Satisfaction", line=dict(color="orange")),
    row=1, col=1
)

fig3.add_trace(
    go.Scatter(x=weeks, y=sd_stockouts, name="Stockouts", line=dict(color="red")),
    row=1, col=1
)

fig3.add_trace(
    go.Scatter(x=weeks, y=sd_capacity, name="Supplier Capacity", line=dict(color="green")),
    row=2, col=1
)

fig3.add_trace(
    go.Scatter(x=weeks, y=customer_demand, name="Customer Demand", line=dict(color="blue", dash="dash")),
    row=2, col=1
)

# Add a vertical line at demand increase week on both subplots
fig3.add_vline(x=demand_increase_week, line_width=2, line_dash="dash", line_color="gray", row=1, col=1)
fig3.add_vline(x=demand_increase_week, line_width=2, line_dash="dash", line_color="gray", row=2, col=1)

fig3.update_layout(height=500, title_text="Additional System Dynamics Components")
fig3.update_xaxes(title_text="Week", row=2, col=1)

st.plotly_chart(fig3, use_container_width=True)

# Calculate bullwhip effect
trad_order_variance = np.var(trad_orders)
sd_order_variance = np.var(sd_orders)
demand_variance = np.var(customer_demand)

trad_bullwhip = round(trad_order_variance / demand_variance, 2)
sd_bullwhip = round(sd_order_variance / demand_variance, 2)

# Key insights
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Model Results")
    avg_trad_inventory = round(np.mean(trad_inventory), 1)
    max_trad_stockout = round(np.max(trad_stockouts), 1)
    total_trad_stockouts = round(np.sum(trad_stockouts), 1)
    
    st.write(f"**Average Inventory: {avg_trad_inventory} units**")
    st.write(f"**Maximum Stockout: {max_trad_stockout} units**")
    st.write(f"**Total Stockouts: {total_trad_stockouts} units**")
    st.write(f"**Bullwhip Ratio: {trad_bullwhip}x** (Order variance / Demand variance)")
    st.write("""
    The traditional model responds directly to demand changes with fixed safety stock
    and lead time assumptions. It doesn't capture how customer behavior or supplier
    constraints might influence the system.
    """)

with col2:
    st.subheader("Systems Dynamics Results")
    avg_sd_inventory = round(np.mean(sd_inventory), 1)
    max_sd_stockout = round(np.max(sd_stockouts), 1)
    total_sd_stockouts = round(np.sum(sd_stockouts), 1)
    min_satisfaction = round(min(sd_satisfaction) * 100, 1)
    
    st.write(f"**Average Inventory: {avg_sd_inventory} units**")
    st.write(f"**Maximum Stockout: {max_sd_stockout} units**")
    st.write(f"**Total Stockouts: {total_sd_stockouts} units**")
    st.write(f"**Minimum Customer Satisfaction: {min_satisfaction}%**")
    st.write(f"**Bullwhip Ratio: {sd_bullwhip}x** (Order variance / Demand variance)")
    st.write("""
    The systems dynamics model reveals how delays, feedback loops, and constraints
    create complex behaviors like the bullwhip effect. It shows how changes in
    one part of the system (like stockouts) affect other parts (like customer 
    satisfaction and future demand).
    """)

st.subheader("The Bullwhip Effect and Why It Matters")
st.write("""
The "bullwhip effect" refers to increasing swings in inventory in response to changes in customer demand
as one moves up the supply chain. Even small changes in consumer demand can result in large variations in
orders at the wholesale, distributor, and manufacturer levels.

**Causes revealed by Systems Dynamics:**
1. **Information delays** - Time lags in recognizing and responding to demand changes
2. **Order batching** - Placing orders in larger, less frequent batches to save on transaction costs
3. **Rationing and shortage gaming** - Ordering more when supply is tight
4. **Price fluctuations** - Buying more during promotions or discounts

Traditional models often miss these dynamics, leading to excess inventory, stockouts,
poor customer service, and higher costs throughout the supply chain.
""")

st.markdown("---")
st.caption("Note: This simulation is simplified for demonstration purposes. Real-world supply chains have additional complexity.")
