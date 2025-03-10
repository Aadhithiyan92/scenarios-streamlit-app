import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Logistic Growth Model", layout="wide")

st.title("Logistic Growth: Traditional vs. Systems Dynamics Approaches")
st.write("""
This simple application demonstrates the fundamental difference between traditional modeling
and systems dynamics using logistic growth - a common pattern in many real-world systems.
""")

# Sidebar controls
st.sidebar.header("Model Parameters")
initial_population = st.sidebar.slider("Initial Population", 10, 1000, 100)
carrying_capacity = st.sidebar.slider("Carrying Capacity", 1000, 10000, 5000)
growth_rate = st.sidebar.slider("Growth Rate", 0.1, 1.0, 0.3, 0.1)
time_periods = st.sidebar.slider("Time Periods to Simulate", 20, 100, 50)
intervention_time = st.sidebar.slider("Intervention Time Period", 10, 30, 20)
intervention_strength = st.sidebar.slider("Intervention Strength (%)", 10, 100, 50)

# Systems Dynamics Parameters
st.sidebar.header("Systems Dynamics Settings")
feedback_delay = st.sidebar.slider("Feedback Delay (periods)", 0, 10, 5)
adaptation_rate = st.sidebar.slider("Adaptation Rate", 0.1, 1.0, 0.5, 0.1)

# Create time periods
time = np.arange(0, time_periods)

# Traditional logistic growth model
def traditional_model(initial_pop, carrying_capacity, growth_rate, time_periods, 
                     intervention_time, intervention_strength):
    population = np.zeros(time_periods)
    growth_rates = np.zeros(time_periods)
    
    # Initial condition
    population[0] = initial_pop
    growth_rates[0] = growth_rate
    
    # Simple logistic growth formula: dP/dt = r*P*(1-P/K)
    for t in range(1, time_periods):
        # Apply intervention at the specified time
        if t == intervention_time:
            # Intervention increases or decreases the population directly
            population[t-1] = population[t-1] * (1 + intervention_strength/100)
        
        # Calculate growth rate - remains constant in traditional model
        growth_rates[t] = growth_rate
        
        # Calculate new population using logistic growth equation
        population[t] = population[t-1] + growth_rates[t] * population[t-1] * (1 - population[t-1]/carrying_capacity)
    
    return population, growth_rates

# Systems dynamics logistic growth model with feedback
def systems_dynamics_model(initial_pop, carrying_capacity, growth_rate, time_periods, 
                         intervention_time, intervention_strength, feedback_delay, adaptation_rate):
    population = np.zeros(time_periods)
    growth_rates = np.zeros(time_periods)
    resource_stress = np.zeros(time_periods)
    
    # Initial conditions
    population[0] = initial_pop
    growth_rates[0] = growth_rate
    resource_stress[0] = population[0]/carrying_capacity
    
    # Systems dynamics simulation with feedback loops
    for t in range(1, time_periods):
        # Apply intervention at the specified time
        if t == intervention_time:
            # Intervention increases or decreases the population directly
            population[t-1] = population[t-1] * (1 + intervention_strength/100)
        
        # Calculate resource stress - ratio of population to carrying capacity
        resource_stress[t] = population[t-1]/carrying_capacity
        
        # Calculate growth rate with feedback and delay
        # Growth rate decreases as population approaches carrying capacity
        if t > feedback_delay:
            # Growth rate adjusts based on resource stress with delay
            delayed_stress = resource_stress[t-feedback_delay]
            growth_rates[t] = growth_rates[t-1] * (1 - adaptation_rate * delayed_stress)
        else:
            growth_rates[t] = growth_rates[t-1]
        
        # Ensure growth rate doesn't go negative
        growth_rates[t] = max(0.01, growth_rates[t])
        
        # Calculate new population using logistic growth equation with variable growth rate
        population[t] = population[t-1] + growth_rates[t] * population[t-1] * (1 - population[t-1]/carrying_capacity)
    
    return population, growth_rates, resource_stress

# Run the models
trad_population, trad_growth_rates = traditional_model(
    initial_population, carrying_capacity, growth_rate, time_periods, 
    intervention_time, intervention_strength
)

sd_population, sd_growth_rates, sd_resource_stress = systems_dynamics_model(
    initial_population, carrying_capacity, growth_rate, time_periods,
    intervention_time, intervention_strength, feedback_delay, adaptation_rate
)

# Create DataFrames for plotting
traditional_df = pd.DataFrame({
    'Time': time,
    'Population': trad_population,
    'Growth Rate': trad_growth_rates
})

sd_df = pd.DataFrame({
    'Time': time,
    'Population': sd_population,
    'Growth Rate': sd_growth_rates,
    'Resource Stress': sd_resource_stress
})

# Main content
st.header("The Logistic Growth Model")
st.write("""
The logistic function models growth with limits, common in:
- Population dynamics (as resources become scarce)
- Technology adoption (as market saturates)
- Disease spread (as immunity develops)
- Product sales (as market matures)

The basic equation is: **dP/dt = r×P×(1-P/K)**
where:
- P is the population (or other quantity)
- r is the growth rate
- K is the carrying capacity (maximum sustainable level)
""")

st.header("Comparing Model Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Model")
    st.write("""
    Traditional models typically:
    - Use fixed parameters
    - Ignore feedback effects on growth rates
    - Treat interventions as one-time events
    - Assume response is immediate
    """)
    
    # Create plotly figure for traditional model
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(
        go.Scatter(x=time, y=trad_population, name="Population", line=dict(color="blue", width=3)),
        secondary_y=False,
    )
    
    fig1.add_trace(
        go.Scatter(x=time, y=trad_growth_rates, name="Growth Rate", line=dict(color="red", width=2, dash="dash")),
        secondary_y=True,
    )
    
    fig1.add_hline(y=carrying_capacity, line_width=1, line_dash="dash", line_color="gray")
    fig1.add_annotation(x=0, y=carrying_capacity*1.02, text="Carrying Capacity", showarrow=False, xanchor="left")
    
    # Add a vertical line at intervention time
    fig1.add_vline(x=intervention_time, line_width=2, line_dash="dash", line_color="green")
    fig1.add_annotation(x=intervention_time, y=max(trad_population)*0.5, 
                      text="Intervention", showarrow=False, textangle=-90)
    
    fig1.update_layout(
        title_text="Traditional Model: Population Growth",
        xaxis_title="Time",
    )
    
    fig1.update_yaxes(title_text="Population", secondary_y=False)
    fig1.update_yaxes(title_text="Growth Rate", secondary_y=True)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Growth rate remains constant throughout
    - Predictable S-shaped population curve
    - Direct, immediate response to intervention
    - No adaptation of system parameters over time
    """)

with col2:
    st.subheader("Systems Dynamics Model")
    st.write("""
    Systems dynamics models incorporate:
    - Feedback loops that change parameters over time
    - Delays between cause and effect
    - Adaptive responses to changing conditions
    - Emergent behaviors from system structure
    """)
    
    # Create plotly figure for systems dynamics model
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(x=time, y=sd_population, name="Population", line=dict(color="blue", width=3)),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(x=time, y=sd_growth_rates, name="Growth Rate", line=dict(color="red", width=2, dash="dash")),
        secondary_y=True,
    )
    
    fig2.add_hline(y=carrying_capacity, line_width=1, line_dash="dash", line_color="gray")
    fig2.add_annotation(x=0, y=carrying_capacity*1.02, text="Carrying Capacity", showarrow=False, xanchor="left")
    
    # Add a vertical line at intervention time
    fig2.add_vline(x=intervention_time, line_width=2, line_dash="dash", line_color="green")
    fig2.add_annotation(x=intervention_time, y=max(sd_population)*0.5, 
                      text="Intervention", showarrow=False, textangle=-90)
    
    fig2.update_layout(
        title_text="Systems Dynamics Model: Population Growth with Feedback",
        xaxis_title="Time",
    )
    
    fig2.update_yaxes(title_text="Population", secondary_y=False)
    fig2.update_yaxes(title_text="Growth Rate", secondary_y=True)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Growth rate adapts based on resource stress
    - Delayed feedback creates more complex behavior
    - System exhibits different responses to intervention
    - Potential for emergent, unexpected patterns
    """)

# Systems Dynamics Key Components
st.header("Systems Dynamics: Revealing the Underlying Mechanisms")

st.write("""
The power of systems dynamics lies in revealing how feedback mechanisms affect system behavior.
Below we see how resource stress influences growth rates with delay, creating more realistic patterns:
""")

# Create a chart to show resource stress
fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(x=time, y=sd_resource_stress, name="Resource Stress", line=dict(color="purple", width=3))
)

fig3.add_trace(
    go.Scatter(x=time, y=sd_growth_rates, name="Growth Rate", line=dict(color="red", width=3))
)

# Add a vertical line at intervention time
fig3.add_vline(x=intervention_time, line_width=2, line_dash="dash", line_color="green")
fig3.add_annotation(x=intervention_time, y=max(sd_resource_stress)*0.5, 
                  text="Intervention", showarrow=False, textangle=-90)

# Add vertical line at feedback delay point after intervention
fig3.add_vline(x=intervention_time+feedback_delay, line_width=1, line_dash="dot", line_color="gray")
fig3.add_annotation(x=intervention_time+feedback_delay, y=max(sd_resource_stress)*0.8, 
                  text="Feedback Delay", showarrow=False, textangle=-90)

fig3.update_layout(
    title_text="Resource Stress and Growth Rate Over Time",
    xaxis_title="Time",
    yaxis_title="Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig3, use_container_width=True)

# Key insights
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Model Results")
    final_trad_pop = round(trad_population[-1], 0)
    max_growth_period_trad = np.argmax(np.diff(trad_population))
    
    st.write(f"**Final Population: {final_trad_pop}**")
    st.write(f"**Period of Fastest Growth: {max_growth_period_trad}**")
    st.write("""
    The traditional model shows a smooth S-curve with little response to intervention
    beyond an immediate level change. The growth rate remains constant throughout
    the simulation, regardless of population density or resource constraints.
    """)

with col2:
    st.subheader("Systems Dynamics Results")
    final_sd_pop = round(sd_population[-1], 0)
    max_growth_period_sd = np.argmax(np.diff(sd_population))
    
    st.write(f"**Final Population: {final_sd_pop}**")
    st.write(f"**Period of Fastest Growth: {max_growth_period_sd}**")
    st.write("""
    The systems dynamics model reveals how internal feedback mechanisms change
    the system behavior over time. The growth rate adapts based on resource stress
    with a delay, creating a more realistic pattern with possible overshoots,
    oscillations, or different equilibrium points.
    """)

st.subheader("Why Systems Dynamics Provides Deeper Understanding")
st.write("""
This simple logistic growth example demonstrates why systems dynamics provides deeper insights:

1. **It Reveals Mechanisms, Not Just Patterns**: Traditional models show WHAT will happen (population growth),
   but systems dynamics shows WHY it happens (feedback between resources and growth rates).

2. **It Captures Time Delays**: Real-world systems rarely respond instantly. Systems dynamics models
   incorporate delays between cause and effect, creating more realistic behavior patterns.

3. **It Models Adaptation**: In the real world, parameters like growth rates change over time in response
   to conditions. Systems dynamics models these changing relationships rather than assuming fixed parameters.

4. **It Predicts Emergent Behaviors**: By modeling the underlying structure, systems dynamics can reveal
   counterintuitive behaviors that emerge from feedback loops and delays, such as overshoots, oscillations,
   or regime shifts.
""")

st.markdown("---")
st.caption("This logistic growth model demonstrates fundamental differences between traditional modeling and systems dynamics approaches.")
