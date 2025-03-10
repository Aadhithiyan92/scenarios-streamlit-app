import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Bathtub Model", layout="wide")

st.title("The Bathtub Model: Traditional vs. Systems Dynamics")
st.write("""
This simple application demonstrates the fundamental difference between traditional modeling approaches
and systems dynamics using a bathtub as an intuitive example everyone can understand.
""")

# Sidebar controls
st.sidebar.header("Model Parameters")
inflow_rate = st.sidebar.slider("Water Inflow Rate (liters/minute)", 5, 20, 10)
initial_outflow = st.sidebar.slider("Initial Drain Size (liters/minute)", 5, 15, 8)
time_horizon = st.sidebar.slider("Time (minutes)", 5, 30, 15)
intervention_time = st.sidebar.slider("When We Adjust the Drain (minute)", 2, 10, 5)
new_outflow = st.sidebar.slider("New Drain Size (liters/minute)", 5, 25, 12)

# Systems Dynamics Parameters
drain_delay = st.sidebar.slider("Drain Adjustment Delay (minutes)", 0, 3, 1)

# Create time periods
minutes = np.arange(0, time_horizon + 1)

# Function to simulate traditional model
def traditional_model(inflow, initial_outflow, new_outflow, intervention_time, time_horizon):
    water_level = np.zeros(time_horizon + 1)
    outflow = np.zeros(time_horizon + 1)
    
    # Initial conditions
    water_level[0] = 0  # Start with empty bathtub
    outflow[0] = initial_outflow
    
    # Simple projection
    for t in range(1, time_horizon + 1):
        # Change in outflow occurs immediately at intervention time
        if t >= intervention_time:
            outflow[t] = new_outflow
        else:
            outflow[t] = initial_outflow
        
        # Water level changes based on net flow
        net_flow = inflow - outflow[t]
        water_level[t] = water_level[t-1] + net_flow
        
        # Water level can't go below zero
        if water_level[t] < 0:
            water_level[t] = 0
    
    return water_level, outflow

# Function to simulate systems dynamics model
def systems_dynamics_model(inflow, initial_outflow, new_outflow, intervention_time, time_horizon, drain_delay):
    water_level = np.zeros(time_horizon + 1)
    outflow = np.zeros(time_horizon + 1)
    target_outflow = np.zeros(time_horizon + 1)
    
    # Initial conditions
    water_level[0] = 0  # Start with empty bathtub
    outflow[0] = initial_outflow
    target_outflow[0] = initial_outflow
    
    # Systems dynamics simulation with stocks and flows
    for t in range(1, time_horizon + 1):
        # Decision to change the drain size happens at intervention time
        if t >= intervention_time:
            target_outflow[t] = new_outflow
        else:
            target_outflow[t] = initial_outflow
        
        # Actual outflow changes gradually due to delay in adjusting the drain
        outflow_adjustment = (target_outflow[t] - outflow[t-1]) / max(1, drain_delay)
        outflow[t] = outflow[t-1] + outflow_adjustment
        
        # Water pressure affects outflow (feedback) - higher water level increases outflow
        pressure_effect = 0.1 * max(0, water_level[t-1])  # Simple linear effect
        actual_outflow = outflow[t] + pressure_effect
        
        # Water level changes based on net flow
        net_flow = inflow - actual_outflow
        water_level[t] = water_level[t-1] + net_flow
        
        # Water level can't go below zero
        if water_level[t] < 0:
            water_level[t] = 0
    
    return water_level, outflow, target_outflow

# Run the models
trad_level, trad_outflow = traditional_model(
    inflow_rate, initial_outflow, new_outflow, intervention_time, time_horizon
)

sd_level, sd_outflow, sd_target = systems_dynamics_model(
    inflow_rate, initial_outflow, new_outflow, intervention_time, time_horizon, drain_delay
)

# Create DataFrames for plotting
traditional_df = pd.DataFrame({
    'Minute': minutes,
    'Water Level': trad_level,
    'Outflow': trad_outflow,
    'Inflow': inflow_rate
})

sd_df = pd.DataFrame({
    'Minute': minutes,
    'Water Level': sd_level,
    'Outflow': sd_outflow,
    'Target Outflow': sd_target,
    'Inflow': inflow_rate
})

# Main content
st.header("The Bathtub Analogy")
st.write("""
The bathtub is a classic systems dynamics example:
- **Water inflow** (the faucet) represents inputs to the system
- **Water level** (in the tub) represents accumulation or the "stock"
- **Water outflow** (the drain) represents outputs from the system

This simple system demonstrates fundamental concepts like stocks, flows, delays, and feedback.
""")

st.header("Comparing Model Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Model")
    st.write("""
    Traditional models typically assume:
    - Changes take effect immediately
    - No delays between decision and implementation
    - Fixed relationships between variables
    - No feedback from outputs back to the system
    """)
    
    # Create plotly figure for traditional model
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(
        go.Scatter(x=minutes, y=trad_level, name="Water Level", line=dict(color="blue", width=3)),
        secondary_y=False,
    )
    
    fig1.add_trace(
        go.Scatter(x=minutes, y=[inflow_rate] * len(minutes), name="Inflow Rate", 
                  line=dict(color="green", width=2, dash="dash")),
        secondary_y=True,
    )
    
    fig1.add_trace(
        go.Scatter(x=minutes, y=trad_outflow, name="Outflow Rate", line=dict(color="red", width=2)),
        secondary_y=True,
    )
    
    # Add a vertical line at intervention time
    fig1.add_vline(x=intervention_time, line_width=2, line_dash="dash", line_color="gray")
    fig1.add_annotation(x=intervention_time, y=max(trad_level)*1.1, text="Drain Adjustment", showarrow=False)
    
    fig1.update_layout(
        title_text="Traditional Model: Bathtub Filling and Draining",
        xaxis_title="Time (minutes)",
    )
    
    fig1.update_yaxes(title_text="Water Level (liters)", secondary_y=False)
    fig1.update_yaxes(title_text="Flow Rate (liters/minute)", secondary_y=True)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Drain adjustment happens instantly
    - No relationship between water level and drain effectiveness
    - Perfectly linear relationships
    - No time delays
    """)

with col2:
    st.subheader("Systems Dynamics Model")
    st.write("""
    Systems dynamics modeling recognizes:
    - Changes take time to implement
    - Delays exist between decisions and effects
    - Feedback loops change how systems behave
    - Stocks accumulate over time
    """)
    
    # Create plotly figure for systems dynamics model
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(x=minutes, y=sd_level, name="Water Level", line=dict(color="blue", width=3)),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(x=minutes, y=[inflow_rate] * len(minutes), name="Inflow Rate", 
                  line=dict(color="green", width=2, dash="dash")),
        secondary_y=True,
    )
    
    fig2.add_trace(
        go.Scatter(x=minutes, y=sd_outflow, name="Actual Outflow", line=dict(color="red", width=2)),
        secondary_y=True,
    )
    
    fig2.add_trace(
        go.Scatter(x=minutes, y=sd_target, name="Target Outflow", line=dict(color="purple", width=2, dash="dot")),
        secondary_y=True,
    )
    
    # Add a vertical line at intervention time
    fig2.add_vline(x=intervention_time, line_width=2, line_dash="dash", line_color="gray")
    fig2.add_annotation(x=intervention_time, y=max(sd_level)*1.1, text="Drain Adjustment Decision", showarrow=False)
    
    fig2.update_layout(
        title_text="Systems Dynamics Model: Bathtub with Delays & Feedback",
        xaxis_title="Time (minutes)",
    )
    
    fig2.update_yaxes(title_text="Water Level (liters)", secondary_y=False)
    fig2.update_yaxes(title_text="Flow Rate (liters/minute)", secondary_y=True)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Time delay between deciding to adjust the drain and completing the adjustment
    - Water pressure affects drain performance (feedback)
    - Nonlinear relationships emerge
    - Water level (stock) accumulates over time
    """)

# Key insights
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Model Results")
    final_level = round(trad_level[-1], 1)
    peak_level = round(max(trad_level), 1)
    
    st.write(f"**Peak Water Level: {peak_level} liters**")
    st.write(f"**Final Water Level: {final_level} liters**")
    st.write("""
    The traditional model shows an immediate response to the drain adjustment.
    The relationship between inputs and outputs is direct and predictable.
    """)

with col2:
    st.subheader("Systems Dynamics Results")
    sd_final_level = round(sd_level[-1], 1)
    sd_peak_level = round(max(sd_level), 1)
    
    st.write(f"**Peak Water Level: {sd_peak_level} liters**")
    st.write(f"**Final Water Level: {sd_final_level} liters**")
    st.write("""
    The systems dynamics model shows a delayed response to the drain adjustment.
    Feedback loops between water level and outflow create more complex behavior.
    """)

st.subheader("How This Relates to Real-World Systems")
st.write("""
This simple bathtub example illustrates fundamental principles that apply to complex real-world systems:

1. **Stocks and Flows**: Like water in a bathtub, many things accumulate (carbon in atmosphere, 
   vehicles on roads, knowledge in organizations)

2. **Delays**: Changes don't happen instantly in real systems (policy implementation takes time,
   infrastructure takes years to build, behavior changes gradually)

3. **Feedback Loops**: System outputs often influence inputs (traffic congestion affects route choices,
   prices affect demand, policy outcomes affect future policies)

4. **Understanding vs. Prediction**: Traditional models might predict certain values, but systems dynamics
   explains WHY systems behave as they do and HOW we might change that behavior
""")

st.markdown("---")
st.caption("This simple bathtub model demonstrates why systems dynamics provides deeper understanding than traditional modeling approaches.")
