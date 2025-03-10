import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Traditional vs. Systems Dynamics Models", layout="wide")

st.title("Carbon Pricing Model: Traditional vs. Systems Dynamics Approaches")
st.write("""
This application demonstrates the fundamental differences between traditional modeling approaches
and systems dynamics modeling using a carbon pricing scenario in transportation.
""")

# Sidebar controls
st.sidebar.header("Model Parameters")
initial_carbon_price = st.sidebar.slider("Initial Carbon Price ($/ton)", 10, 100, 30)
time_horizon = st.sidebar.slider("Time Horizon (years)", 5, 30, 15)
policy_change_year = st.sidebar.slider("Policy Change Year", 2, 10, 5)
new_carbon_price = st.sidebar.slider("New Carbon Price After Policy Change ($/ton)", 
                                    initial_carbon_price, 200, initial_carbon_price + 50)

# Traditional Model Parameters
tech_adoption_rate = st.sidebar.slider("Technology Adoption Rate (%/year)", 1.0, 10.0, 3.0)
price_elasticity = st.sidebar.slider("Price Elasticity of Emissions", -1.0, -0.1, -0.3, 0.1)

# Systems Dynamics Parameters
tech_learning_rate = st.sidebar.slider("Technology Learning Rate (%)", 5, 25, 15)
behavioral_adaptation = st.sidebar.slider("Behavioral Adaptation Factor", 0.1, 2.0, 0.5, 0.1)
investment_delay = st.sidebar.slider("Investment Delay (years)", 1, 5, 2)

# Create time periods
years = np.arange(1, time_horizon + 1)

# Function to simulate traditional econometric model
def traditional_model(carbon_price, time_periods, price_elasticity, tech_adoption_rate, policy_year, new_price):
    emissions = np.zeros(time_periods)
    prices = np.zeros(time_periods)
    
    # Initial conditions
    base_emissions = 100  # Arbitrary base emissions
    emissions[0] = base_emissions
    prices[0] = carbon_price
    
    # Simple projection based on carbon price
    for t in range(1, time_periods):
        # Policy change
        if t >= policy_year - 1:
            prices[t] = new_price
        else:
            prices[t] = prices[t-1]
        
        # Linear reduction from technology
        tech_factor = 1 - (tech_adoption_rate/100) * t
        
        # Price elasticity effect
        price_factor = (prices[t] / prices[0]) ** price_elasticity
        
        # Combined effect
        emissions[t] = base_emissions * tech_factor * price_factor
    
    return emissions, prices

# Function to simulate systems dynamics model
def systems_dynamics_model(carbon_price, time_periods, price_elasticity, tech_learning_rate, 
                          behavioral_adaptation, investment_delay, policy_year, new_price):
    # Initialize arrays
    emissions = np.zeros(time_periods)
    prices = np.zeros(time_periods)
    technology_level = np.zeros(time_periods)
    investment = np.zeros(time_periods)
    cumulative_investment = np.zeros(time_periods)
    behavioral_change = np.zeros(time_periods)
    
    # Initial conditions
    base_emissions = 100  # Same base as traditional model
    emissions[0] = base_emissions
    prices[0] = carbon_price
    technology_level[0] = 1.0  # Initial technology efficiency (normalized)
    investment[0] = 0.1 * carbon_price  # Initial investment as function of price
    cumulative_investment[0] = investment[0]
    behavioral_change[0] = 1.0  # No initial behavioral change
    
    # Systems dynamics simulation with feedback loops
    for t in range(1, time_periods):
        # Policy change
        if t >= policy_year - 1:
            prices[t] = new_price
        else:
            prices[t] = prices[t-1]
        
        # Investment responds to carbon price with delay
        if t >= investment_delay:
            investment[t] = 0.1 * prices[t-investment_delay] * (1 + behavioral_change[t-1] * 0.5)
        else:
            investment[t] = 0.1 * prices[t] * (1 + behavioral_change[t-1] * 0.5)
        
        # Cumulative investment drives technology learning
        cumulative_investment[t] = cumulative_investment[t-1] + investment[t]
        
        # Technology improves based on cumulative investment (learning curve)
        technology_level[t] = technology_level[t-1] * (1 - tech_learning_rate/100 * 
                                                      np.log(cumulative_investment[t]/cumulative_investment[t-1] + 1))
        
        # Behavioral change responds to price signal and visible technology improvements
        behavioral_change[t] = behavioral_change[t-1] * (1 + behavioral_adaptation * 
                                                       (prices[t]/prices[t-1] - 1) * 
                                                       (technology_level[t-1]/technology_level[t]))
        
        # Combined effects on emissions with feedback
        emissions[t] = emissions[t-1] * technology_level[t] / technology_level[t-1] * behavioral_change[t]
    
    return emissions, prices, technology_level, investment, behavioral_change

# Run the models
traditional_emissions, traditional_prices = traditional_model(
    initial_carbon_price, time_horizon, price_elasticity, tech_adoption_rate, 
    policy_change_year, new_carbon_price
)

sd_emissions, sd_prices, tech_level, investment, behavior = systems_dynamics_model(
    initial_carbon_price, time_horizon, price_elasticity, tech_learning_rate,
    behavioral_adaptation, investment_delay, policy_change_year, new_carbon_price
)

# Create DataFrames for plotting
traditional_df = pd.DataFrame({
    'Year': years,
    'Carbon Price': traditional_prices,
    'Emissions': traditional_emissions
})

sd_df = pd.DataFrame({
    'Year': years,
    'Carbon Price': sd_prices,
    'Emissions': sd_emissions,
    'Technology Level': tech_level,
    'Investment': investment,
    'Behavioral Change': behavior
})

# Main content
st.header("Comparing Model Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Econometric Model")
    st.write("""
    Traditional models typically use elasticities and linear trends to predict how
    emissions respond to carbon prices. They often miss complex interactions and feedback loops.
    """)
    
    # Create plotly figure for traditional model
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(
        go.Scatter(x=years, y=traditional_emissions, name="Emissions", line=dict(color="red", width=3)),
        secondary_y=False,
    )
    
    fig1.add_trace(
        go.Scatter(x=years, y=traditional_prices, name="Carbon Price", line=dict(color="blue", width=3, dash="dash")),
        secondary_y=True,
    )
    
    fig1.update_layout(
        title_text="Traditional Model: Emissions vs Carbon Price",
        xaxis_title="Year",
    )
    
    fig1.update_yaxes(title_text="Emissions (tons)", secondary_y=False)
    fig1.update_yaxes(title_text="Carbon Price ($/ton)", secondary_y=True)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Linear relationships between variables
    - Static response to price changes
    - No emerging behaviors or feedback effects
    - Missing delayed responses and adaptation
    """)

with col2:
    st.subheader("Systems Dynamics Model")
    st.write("""
    Systems dynamics captures feedback loops, delays, and non-linear relationships
    in how the transportation system responds to carbon pricing over time.
    """)
    
    # Create plotly figure for systems dynamics model
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig2.add_trace(
        go.Scatter(x=years, y=sd_emissions, name="Emissions", line=dict(color="red", width=3)),
        secondary_y=False,
    )
    
    fig2.add_trace(
        go.Scatter(x=years, y=sd_prices, name="Carbon Price", line=dict(color="blue", width=3, dash="dash")),
        secondary_y=True,
    )
    
    fig2.update_layout(
        title_text="Systems Dynamics Model: Emissions vs Carbon Price",
        xaxis_title="Year",
    )
    
    fig2.update_yaxes(title_text="Emissions (tons)", secondary_y=False)
    fig2.update_yaxes(title_text="Carbon Price ($/ton)", secondary_y=True)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.write("""
    **Key Characteristics:**
    - Feedback loops between price, technology, and behavior
    - Time delays in investment and technology adoption
    - Non-linear responses that change over time
    - Emergent behaviors not apparent in the inputs
    """)

# Systems Dynamics Key Components
st.header("Systems Dynamics: Revealing the Underlying Structure")

st.write("""
The power of systems dynamics lies in revealing the causal relationships and feedback loops
that drive system behavior. Below we see how technology, investment, and behavioral changes
interact over time in response to carbon pricing.
""")

# Create 3-panel chart to show key drivers in systems dynamics model
fig3 = make_subplots(rows=3, cols=1, 
                    subplot_titles=("Technology Improvement", "Investment Response", "Behavioral Adaptation"))

fig3.add_trace(
    go.Scatter(x=years, y=tech_level, name="Technology Level", line=dict(color="green")),
    row=1, col=1
)

fig3.add_trace(
    go.Scatter(x=years, y=investment, name="Investment", line=dict(color="purple")),
    row=2, col=1
)

fig3.add_trace(
    go.Scatter(x=years, y=behavior, name="Behavioral Change", line=dict(color="orange")),
    row=3, col=1
)

fig3.update_layout(height=600, title_text="Key Drivers in the Systems Dynamics Model")
fig3.update_xaxes(title_text="Year", row=3, col=1)

st.plotly_chart(fig3, use_container_width=True)

# Key insights
st.header("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Traditional Model Results")
    diff_trad = round(((traditional_emissions[-1] / traditional_emissions[0]) - 1) * 100, 1)
    st.write(f"**Final Emissions Change: {diff_trad}%**")
    st.write(f"**Carbon Price in Year {time_horizon}: ${traditional_prices[-1]:.2f}/ton**")
    st.write("""
    The traditional model shows a relatively straightforward relationship between
    carbon price and emissions reduction. It assumes a constant rate of technology
    adoption and a fixed elasticity of response to price changes.
    """)

with col2:
    st.subheader("Systems Dynamics Results")
    diff_sd = round(((sd_emissions[-1] / sd_emissions[0]) - 1) * 100, 1)
    st.write(f"**Final Emissions Change: {diff_sd}%**")
    st.write(f"**Carbon Price in Year {time_horizon}: ${sd_prices[-1]:.2f}/ton**")
    st.write("""
    The systems dynamics model reveals how the same carbon price can lead to
    different outcomes due to reinforcing feedback loops, delayed investments,
    and non-linear technology learning curves. These dynamics are crucial for
    effective policy design but are typically missed by traditional models.
    """)

st.subheader("Why These Differences Matter for Policy")
st.write("""
1. **Policy Design**: Systems dynamics suggests that the timing and structure of carbon pricing
   matters as much as the price level itself.

2. **Implementation Strategy**: Understanding delays and feedback loops helps design more
   effective implementation strategies that account for system inertia.

3. **Intervention Points**: Systems dynamics reveals high-leverage intervention points that
   may not be apparent in traditional models.

4. **Unexpected Outcomes**: The systems approach helps anticipate unintended consequences and
   counterintuitive system behaviors that traditional models often miss.
""")

st.markdown("---")
st.caption("Note: This simulation is simplified for demonstration purposes. Real-world models would include additional variables and interactions.")