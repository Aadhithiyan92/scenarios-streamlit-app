import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Highway Congestion Model", layout="wide")

st.title("Highway Expansion Paradox: Traditional vs. Systems Dynamics Approaches")
st.write("""
This application demonstrates the fundamental differences between traditional modeling approaches
and systems dynamics modeling using the well-known "induced demand" phenomenon in transportation.
""")

# Sidebar controls
st.sidebar.header("Model Parameters")
initial_lanes = st.sidebar.slider("Initial Number of Highway Lanes", 2, 6, 3)
time_horizon = st.sidebar.slider("Time Horizon (years)", 5, 20, 10)
expansion_year = st.sidebar.slider("Highway Expansion Year", 2, 5, 3)
lane_increase = st.sidebar.slider("Number of Lanes Added", 1, 3, 2)

# Traditional Model Parameters
traffic_growth_rate = st.sidebar.slider("Annual Traffic Growth Rate (%)", 1.0, 10.0, 3.0)
capacity_per_lane = st.sidebar.slider("Capacity Per Lane (vehicles/hour)", 1000, 2500, 2000)

# Systems Dynamics Parameters
induced_demand_factor = st.sidebar.slider("Induced Demand Factor", 0.0, 2.0, 1.0, 0.1)
land_use_delay = st.sidebar.slider("Land Use Change Delay (years)", 1, 5, 2)
transit_mode_share = st.sidebar.slider("Initial Transit Mode Share (%)", 5, 30, 15)

# Create time periods
years = np.arange(1, time_horizon + 1)

# Function to simulate traditional traffic model
def traditional_model(initial_lanes, expansion_year, lane_increase, time_horizon, traffic_growth_rate, capacity_per_lane):
    capacity = np.zeros(time_horizon)
    traffic_volume = np.zeros(time_horizon)
    congestion_ratio = np.zeros(time_horizon)
    travel_time = np.zeros(time_horizon)
    lanes = np.zeros(time_horizon)
    
    # Initial conditions
    lanes[0] = initial_lanes
    capacity[0] = initial_lanes * capacity_per_lane
    traffic_volume[0] = capacity[0] * 0.8  # Starting at 80% capacity
    congestion_ratio[0] = traffic_volume[0] / capacity[0]
    travel_time[0] = 30  # Base travel time in minutes
    
    # Simple projection
    for t in range(1, time_horizon):
        # Lane expansion
        if t >= expansion_year - 1:
            lanes[t] = initial_lanes + lane_increase
        else:
            lanes[t] = lanes[t-1]
        
        # Update capacity
        capacity[t] = lanes[t] * capacity_per_lane
        
        # Traffic grows at fixed rate, regardless of capacity
        traffic_volume[t] = traffic_volume[0] * (1 + traffic_growth_rate/100) ** t
        
        # Calculate congestion and travel time
        congestion_ratio[t] = traffic_volume[t] / capacity[t]
        
        # Simple travel time model - exponential increase when congestion ratio > 0.8
        if congestion_ratio[t] > 0.8:
            travel_time[t] = 30 * (1 + 2 * (congestion_ratio[t] - 0.8) ** 2)
        else:
            travel_time[t] = 30
    
    return lanes, capacity, traffic_volume, congestion_ratio, travel_time

# Function to simulate systems dynamics model with induced demand
def systems_dynamics_model(initial_lanes, expansion_year, lane_increase, time_horizon, traffic_growth_rate, 
                          capacity_per_lane, induced_demand_factor, land_use_delay, transit_mode_share):
    capacity = np.zeros(time_horizon)
    traffic_volume = np.zeros(time_horizon)
    congestion_ratio = np.zeros(time_horizon)
    travel_time = np.zeros(time_horizon)
    lanes = np.zeros(time_horizon)
    land_use_intensity = np.zeros(time_horizon)
    transit_share = np.zeros(time_horizon)
    
    # Initial conditions
    lanes[0] = initial_lanes
    capacity[0] = initial_lanes * capacity_per_lane
    traffic_volume[0] = capacity[0] * 0.8  # Starting at 80% capacity
    congestion_ratio[0] = traffic_volume[0] / capacity[0]
    travel_time[0] = 30  # Base travel time in minutes
    land_use_intensity[0] = 1.0  # Normalized initial land use intensity
    transit_share[0] = transit_mode_share / 100  # Convert to decimal
    
    # Systems dynamics simulation with feedback loops
    for t in range(1, time_horizon):
        # Lane expansion
        if t >= expansion_year - 1:
            lanes[t] = initial_lanes + lane_increase
        else:
            lanes[t] = lanes[t-1]
        
        # Update capacity
        capacity[t] = lanes[t] * capacity_per_lane
        
        # Travel time based on previous period's congestion
        if congestion_ratio[t-1] > 0.8:
            travel_time[t] = 30 * (1 + 2 * (congestion_ratio[t-1] - 0.8) ** 2)
        else:
            travel_time[t] = 30
        
        # Land use responds to travel time with delay
        if t > land_use_delay:
            # If travel time decreased compared to previous periods, land use intensity increases
            travel_time_change = travel_time[t-land_use_delay] / travel_time[max(0, t-land_use_delay-1)] - 1
            land_use_intensity[t] = land_use_intensity[t-1] * (1 - induced_demand_factor * travel_time_change)
        else:
            land_use_intensity[t] = land_use_intensity[t-1]
        
        # Transit share responds to road capacity and congestion
        # More road capacity tends to reduce transit use, while congestion increases it
        capacity_change = capacity[t] / capacity[t-1] - 1
        transit_effect = -0.1 * capacity_change + 0.05 * (congestion_ratio[t-1] - 0.8) if congestion_ratio[t-1] > 0.8 else -0.1 * capacity_change
        transit_share[t] = max(0.05, min(0.5, transit_share[t-1] * (1 + transit_effect)))
        
        # Base traffic growth plus induced demand effects
        base_growth = traffic_volume[0] * (1 + traffic_growth_rate/100) ** t
        induced_effect = 1.0 + (capacity[t] / capacity[max(0, t-1)] - 1) * induced_demand_factor
        land_use_effect = land_use_intensity[t] / land_use_intensity[max(0, t-1)]
        
        # Calculate traffic volume with feedback effects
        traffic_volume[t] = base_growth * induced_effect * land_use_effect * (1 - transit_share[t]) / (1 - transit_share[0])
        
        # Calculate congestion ratio
        congestion_ratio[t] = traffic_volume[t] / capacity[t]
    
    return lanes, capacity, traffic_volume, congestion_ratio, travel_time, land_use_intensity, transit_share

# Run the models
trad_lanes, trad_capacity, trad_traffic, trad_congestion, trad_time = traditional_model(
    initial_lanes, expansion_year, lane_increase, time_horizon, traffic_growth_rate, capacity_per_lane
)

sd_lanes, sd_capacity, sd_traffic, sd_congestion, sd_time, sd_land_use, sd_transit = systems_dynamics_model(
    initial_lanes, expansion_year, lane_increase, time_horizon, traffic_growth_rate,
    capacity_per_lane, induced_demand_factor, land_use_delay, transit_mode_share
)

# Create DataFrames for plotting
traditional_df = pd.DataFrame({
    'Year': years,
    'Lanes': trad_lanes,
    'Capacity': trad_capacity,
    'Traffic Volume': trad_traffic,
    'Congestion Ratio': trad_congestion,
    'Travel Time': trad_time
})

sd_df = pd.DataFrame({
    'Year': years,
    'Lanes': sd_lanes,
    'Capacity': sd_capacity,
    'Traffic Volume': sd_traffic,
    'Congestion Ratio': sd_congestion,
    'Travel Time': sd_time,
    'Land Use Intensity': sd_land_use,
    'Transit Share': sd_transit
})

# Main content
st.header("Comparing Model Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Transportation Model")
    st.write("""
    Traditional models typically assume traffic grows at a fixed rate independent of
    road capacity. They predict that adding lanes directly reduces congestion and travel time.
    """)
    
    # Create plotly figure for traditional model
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(
        go.Scatter(x=years, y=trad_traffic, name="Traffic Volume", line=dict(color="blue", width=3)),
        secondary_y=False,
    )
    
    fig1.add_trace(
        go.Scatter(x=years, y=trad_capacity, name="Road Capacity", line=dict(color="green", width=3, dash="dash")),
        secondary_y=False,
    )
    
    fig1.add_trace(
        go.Scatter(x=years, y=trad_time, name="Travel Time", line=dict(color="red", width=3)),
        secondary_y=True,
    )
    
    # Add a vertical line at expansion year
    fig1.add_vline(x=expansion_year, line_width=2, line_dash="dash", line_color="gray")
    fig1.add_annotation(x=expansion_year, y=max(trad_capacity)*1.1, text="Lane Expansion", showarrow=False)
    
    fig1.update_layout(
        title_text="Traditional Model: Traffic vs Capacity",
        xaxis_title="Year",
    )
    
    fig1.update_yaxes(title_text="Volume/Capacity (vehicles/hour)", secondary_y=False)
    fig1.update_yaxes(title_text="Travel Time (minutes)", secondary_y=True)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Traffic growth is independent of road capacity
    - Direct linear relationship between capacity and congestion
    - Adding lanes always reduces travel time
    - No feedback effects between infrastructure and demand
    """)

with col2:
    st.subheader("Systems Dynamics Model")
    st.write("""
    Systems dynamics captures "induced demand" - the phenomenon where expanding roads
    actually increases traffic, often negating congestion benefits. This occurs through
    multiple feedback loops in the transportation system.
    """)
    
    # Create plotly figure for systems dynamics model
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(x=years, y=sd_traffic, name="Traffic Volume", line=dict(color="blue", width=3)),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(x=years, y=sd_capacity, name="Road Capacity", line=dict(color="green", width=3, dash="dash")),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(x=years, y=sd_time, name="Travel Time", line=dict(color="red", width=3)),
        secondary_y=True,
    )
    
    # Add a vertical line at expansion year
    fig2.add_vline(x=expansion_year, line_width=2, line_dash="dash", line_color="gray")
    fig2.add_annotation(x=expansion_year, y=max(sd_capacity)*1.1, text="Lane Expansion", showarrow=False)
    
    fig2.update_layout(
        title_text="Systems Dynamics Model: Traffic vs Capacity",
        xaxis_title="Year",
    )
    
    fig2.update_yaxes(title_text="Volume/Capacity (vehicles/hour)", secondary_y=False)
    fig2.update_yaxes(title_text="Travel Time (minutes)", secondary_y=True)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Traffic volume responds to capacity changes (induced demand)
    - Time delays between infrastructure changes and full effects
    - Feedback loops between travel time, land use, and mode choice
    - Non-linear relationships that change over time
    """)

# Systems Dynamics Key Components
st.header("Systems Dynamics: Revealing the Underlying Structure")

st.write("""
The power of systems dynamics lies in revealing how feedback loops create counterintuitive
system behavior. Below we see how land use patterns and transit usage respond to highway
expansion, ultimately affecting traffic congestion.
""")

# Create 2-panel chart to show key drivers in systems dynamics model
fig3 = make_subplots(rows=2, cols=1, 
                    subplot_titles=("Land Use Intensity Response", "Transit Mode Share"))

fig3.add_trace(
    go.Scatter(x=years, y=sd_land_use, name="Land Use Intensity", line=dict(color="purple")),
    row=1, col=1
)

fig3.add_trace(
    go.Scatter(x=years, y=sd_transit, name="Transit Mode Share", line=dict(color="orange")),
    row=2, col=1
)

# Add a vertical line at expansion year on both subplots
fig3.add_vline(x=expansion_year, line_width=2, line_dash="dash", line_color="gray", row=1, col=1)
fig3.add_vline(x=expansion_year, line_width=2, line_dash="dash", line_color="gray", row=2, col=1)

fig3.update_layout(height=500, title_text="Key Feedback Mechanisms in the Systems Dynamics Model")
fig3.update_xaxes(title_text="Year", row=2, col=1)

st.plotly_chart(fig3, use_container_width=True)

# Key insights
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Model Results")
    congestion_change = round(((trad_congestion[-1] / trad_congestion[expansion_year-1]) - 1) * 100, 1)
    travel_time_change = round(((trad_time[-1] / trad_time[expansion_year-1]) - 1) * 100, 1)
    
    st.write(f"**Congestion Ratio Change After Expansion: {congestion_change}%**")
    st.write(f"**Travel Time Change After Expansion: {travel_time_change}%**")
    st.write("""
    The traditional model predicts that adding lanes directly reduces congestion
    and improves travel times. It assumes traffic growth is independent of road 
    capacity, leading to linear and predictable improvements.
    """)

with col2:
    st.subheader("Systems Dynamics Results")
    sd_congestion_change = round(((sd_congestion[-1] / sd_congestion[expansion_year-1]) - 1) * 100, 1)
    sd_travel_time_change = round(((sd_time[-1] / sd_time[expansion_year-1]) - 1) * 100, 1)
    
    st.write(f"**Congestion Ratio Change After Expansion: {sd_congestion_change}%**")
    st.write(f"**Travel Time Change After Expansion: {sd_travel_time_change}%**")
    st.write("""
    The systems dynamics model reveals how adding lanes can counterintuitively 
    increase congestion through induced demand. Initial travel time improvements
    trigger changes in land use and mode choice that ultimately generate more traffic,
    potentially negating the benefits of expansion.
    """)

st.subheader("Why These Differences Matter for Transportation Policy")
st.write("""
1. **The Induced Demand Paradox**: Systems dynamics explains why highway expansion often fails
   to reduce congestion long-term - a finding consistently observed in real-world studies.

2. **Policy Design**: Understanding feedback loops helps design more effective transportation
   policies that consider both infrastructure and demand management.

3. **Integrated Planning**: The systems approach highlights the importance of coordinating
   transportation infrastructure with land use planning and transit investments.

4. **Long-term Outcomes**: Traditional models often overestimate the benefits of highway expansion
   by failing to account for the systemic responses that occur over time.
""")

st.markdown("---")
st.caption("Note: This simulation is simplified for demonstration purposes. Real-world models would include additional variables and interactions.")
