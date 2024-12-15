import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(layout="wide", page_title="BEV Adoption Strategy Analysis")

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel('C:/Users/as3889/Desktop/ZEVdata.xlsx')
    incentive_data = pd.DataFrame({
        'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        'CVRP_Amount': [375000, 15156388, 20103380, 58796855, 90102019, 102848457, 101956078, 117563328,
                        173909297, 176018763, 95876302, 125864437, 101382166, 331899666],
        'CVAP_Amount': [0, 0, 0, 0, 0, 0, 0, 0, 2002253, 10000, 19418653, 1520500, 15000, 2214500],
        'DCAP_Amount': [0, 0, 0, 0, 0, 0, 10000, 45000, 190000, 245000, 336500, 544500, 581000, 689500],
        'CC4A_Amount': [0, 0, 0, 0, 0, 1898814, 4922899, 6068746, 10275857, 20377745, 23440151, 13864693,
                        13534438, 20614737]
    })
    return data, incentive_data

data, incentive_data = load_data()

years = data['Year'].values
V_ICE_data = data['V'].values
V_BEV_data = data['BEV'].values
M_data = data['M'].values
C_data = data['C'].values
S_data = data['S'].values

# Normalize data
V_ICE_mean = np.mean(V_ICE_data)
V_BEV_mean = np.mean(V_BEV_data)
M_mean = np.mean(M_data)
C_mean = np.mean(C_data)
S_mean = np.mean(S_data)

V_ICE_norm = V_ICE_data / V_ICE_mean
V_BEV_norm = V_BEV_data / V_BEV_mean
M_norm = M_data / M_mean
C_norm = C_data / C_mean
S_norm = S_data / S_mean

# Normalize incentive data
mean_incentives = {
    'CVRP': np.mean(incentive_data['CVRP_Amount'][incentive_data['CVRP_Amount'] > 0]),
    'CVAP': np.mean(incentive_data['CVAP_Amount'][incentive_data['CVAP_Amount'] > 0]),
    'DCAP': np.mean(incentive_data['DCAP_Amount'][incentive_data['DCAP_Amount'] > 0]),
    'CC4A': np.mean(incentive_data['CC4A_Amount'][incentive_data['CC4A_Amount'] > 0])
}

for prog in ['CVRP', 'CVAP', 'DCAP', 'CC4A']:
    incentive_data[f'{prog}_Norm'] = np.where(
        incentive_data[f'{prog}_Amount'] > 0,
        incentive_data[f'{prog}_Amount'] / mean_incentives[prog],
        0
    )

# Parameters
params = {
    'r1': 0.04600309537728457,
    'K1': 19.988853430783674,
    'alpha1': 2.1990994330017567e-20,
    'r2': 0.4242847858514477,
    'beta1': 0.0010,
    'gamma1': 0.006195813413796029,
    'phi1': 2.420439212993679,
    'phi2': 0.03290625472523172,
    'eta': 2.472995227365455,
    'psi1': 0.877999000323586,
    'psi2': 0.8825501230310412,
    'delta': 0.8114852069566516,
    'epsilon': 1.6113341002621433e-34,
    'zeta': 8.756752159763104e-12,
    'k_C': 9.466080903837223,
    'k_V': 1.1743857370252369,
    'k_A': 0.9252378762028231,
    'k_D': 1.282707168138512,
    'lambda_C': 0.1699545021325927,
    'lambda_V': 0.16153300196227824,
    'lambda_A': 0.16300789498583168,
    'lambda_D': 0.16186856013035358,
    'kappa': 0.3673510495799293,
    'lambda_S': 0.004719196117653555,
    'omega': 0.0773179311036966,
    'tau': 0.04,
    'cvrp_end_year': 2027
}

def incentive_time_effect(t, start_year, end_year, incentive_type, decay_rate):
    """Calculate incentive effect at a given time"""
    year = t + 2010  # Start from 2010

    if year in incentive_data['Year'].values:
        return incentive_data.loc[incentive_data['Year'] == year, f'{incentive_type}_Norm'].values[0]

    if start_year <= year <= end_year:
        return np.exp(-decay_rate * (year - start_year))
    return 0

def system(t, X, params):
    """System of differential equations for BEV adoption"""
    V, B, M, C, S = X

    year = t + 2010  # Start from 2010
    if year in incentive_data['Year'].values:
        idx = incentive_data['Year'] == year
        cvrp_effect = incentive_data.loc[idx, 'CVRP_Norm'].values[0]
        cvap_effect = incentive_data.loc[idx, 'CVAP_Norm'].values[0]
        cc4a_effect = incentive_data.loc[idx, 'CC4A_Norm'].values[0]
        dcap_effect = incentive_data.loc[idx, 'DCAP_Norm'].values[0]
    else:
        cvrp_effect = incentive_time_effect(t, 2010, params['cvrp_end_year'], 'CVRP', params['lambda_C'])
        cvap_effect = incentive_time_effect(t, 2010, params['cvrp_end_year'], 'CVAP', params['lambda_V'])
        cc4a_effect = incentive_time_effect(t, 2010, params['cvrp_end_year'], 'CC4A', params['lambda_A'])
        dcap_effect = incentive_time_effect(t, 2010, params['cvrp_end_year'], 'DCAP', params['lambda_D'])

    total_incentive = (params['k_C'] * cvrp_effect +
                      params['k_V'] * cvap_effect +
                      params['k_A'] * cc4a_effect +
                      params['k_D'] * dcap_effect)

    total_vehicles = V + B
    ev_fraction = B / total_vehicles if total_vehicles > 0 else 0

    # System equations
    dV_dt = params['r1'] * V * (1 - total_vehicles/params['K1']) * (1 - params['omega'] * ev_fraction) - params['tau'] * V * ev_fraction - params['epsilon'] * V
    dB_dt = params['r2'] * B + params['beta1'] * total_incentive + params['alpha1'] * params['tau'] * V * ev_fraction - params['gamma1'] * B
    dM_dt = params['phi1'] * V + params['phi2'] * B - params['eta'] * M
    dC_dt = (params['psi1'] * V + params['psi2'] * B) * M / total_vehicles - params['delta'] * C + params['zeta'] * (V / total_vehicles) ** 2 if total_vehicles > 0 else 0
    dS_dt = params['kappa'] * B / total_vehicles - params['lambda_S'] * S if total_vehicles > 0 else 0

    return [dV_dt, dB_dt, dM_dt, dC_dt, dS_dt]

def run_projection(base_params, total_years=17):
    """Run a single projection from 2010 to 2027"""
    projection_params = base_params.copy()

    # Initial conditions from 2010
    X0_2010 = [
        V_ICE_data[0]/V_ICE_mean,
        V_BEV_data[0]/V_BEV_mean,
        M_data[0]/M_mean,
        C_data[0]/C_mean,
        S_data[0]/S_mean
    ]

    # Solve from 2010 to 2027
    t_eval = np.linspace(0, total_years, total_years*12+1)  # Monthly resolution
    solution = solve_ivp(
        system,
        (0, total_years),
        X0_2010,
        args=(projection_params,),
        t_eval=t_eval,
        method='RK45'
    )

    bev = solution.y[1] * V_BEV_mean
    years_proj = np.array([2010 + t for t in solution.t])

    return {
        'years': years_proj,
        'bev': bev
    }

# Run projection
projection_results = run_projection(params, total_years=17)  # 2010 to 2027

st.title("Strategic Analysis of BEV Adoption in California (2010-2027)")

st.markdown("""
### Executive Summary
This analysis projects the adoption of Battery Electric Vehicles (BEV) in California from 2010 to 2027, utilizing historical data up to 2023 for model validation.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["BEV Adoption Projection", "Impact Analysis", "Recommendations"])

with tab1:
    st.header("BEV Adoption Trajectory")
    
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=years,
        y=V_BEV_data,
        name="Historical BEV Data",
        mode='markers+lines',
        line=dict(width=2, dash='dash'),
        marker=dict(color='blue')
    ))
    
    # Plot projection
    fig.add_trace(go.Scatter(
        x=projection_results['years'],
        y=projection_results['bev'],
        name="Projected BEV Adoption",
        mode='lines',
        line=dict(width=3, color='green')
    ))
    
    # Add vertical line at 2024
    fig.add_vline(x=2024, line_dash="dash", line_color="gray", 
                 annotation_text="Projection Start", annotation_position="top left")
    
    fig.update_layout(
        title='BEV Adoption Trajectory (2010-2027)',
        xaxis_title='Year',
        yaxis_title='BEV Vehicles',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add metrics
    st.subheader("Key Metrics")
    final_adoption = projection_results['bev'][-1]
    historical_adoption = V_BEV_data[-1]
    improvement = (final_adoption - historical_adoption) / historical_adoption * 100
    metrics = pd.DataFrame({
        'Metric': ['Final BEV Adoption (2027)', 'Improvement from 2023 (%)'],
        'Value': [f"{final_adoption:.0f}", f"{improvement:.1f}%"]
    })
    st.table(metrics)

with tab2:
    st.header("Impact Analysis")
    
    # Calculate annual adoption
    proj_df = pd.DataFrame({
        'Year': projection_results['years'],
        'BEV_Vehicles': projection_results['bev']
    })
    
    # Calculate year-over-year growth
    proj_df['Yearly_Growth'] = proj_df['BEV_Vehicles'].pct_change() * 100
    
    # Plot yearly growth
    fig_growth = px.line(
        proj_df,
        x='Year',
        y='Yearly_Growth',
        title="Year-over-Year BEV Adoption Growth",
        labels={'Year': 'Year', 'Yearly_Growth': 'Growth (%)'},
        markers=True
    )
    st.plotly_chart(fig_growth, use_container_width=True)
    
    st.markdown("""
    ### Analysis Insights
    - **Consistent Growth**: The projection indicates a steady increase in BEV adoption, driven by continued incentives and market dynamics.
    - **Policy Impact**: Ongoing incentive programs play a crucial role in sustaining growth rates.
    - **Market Saturation**: As adoption increases, growth rates may stabilize, reflecting market saturation and technological advancements.
    """)

with tab3:
    st.header("Strategic Recommendations")
    
    st.markdown(f"""
    ### Key Findings
    1. **Steady Growth Trajectory**:
        - The BEV market in California is projected to grow consistently from 2010 to 2027.
        - By 2027, BEV adoption is expected to reach approximately **{projection_results['bev'][-1]:.0f} vehicles**.
    
    2. **Effective Incentive Programs**:
        - Incentive programs have a significant impact on accelerating BEV adoption.
        - Continued and enhanced incentives can sustain and boost growth rates.
    
    3. **Infrastructure Development**:
        - Expansion of charging infrastructure is essential to support increasing BEV numbers.
        - Public-private partnerships can facilitate infrastructure investments.
    
    ### Recommended Implementation Strategy
    1. **Phase 1 (2024-2025)**:
        - **Enhance Incentive Programs**: Increase financial incentives for both consumers and manufacturers.
        - **Expand Charging Infrastructure**: Focus on underserved areas to ensure widespread accessibility.
    
    2. **Phase 2 (2026-2027)**:
        - **Sustain Growth Initiatives**: Maintain and adjust incentives based on market response.
        - **Promote Technological Innovation**: Support advancements in battery technology and vehicle efficiency.
    
    ### Critical Success Factors
    - **Consistent Policy Support**: Stable and predictable incentives to encourage long-term investments.
    - **Stakeholder Collaboration**: Engage with automakers, energy providers, and communities to align efforts.
    - **Market Monitoring**: Regularly assess market trends and adjust strategies accordingly.
    - **Public Awareness Campaigns**: Educate consumers on the benefits and availability of BEVs.
    """)

# Add sidebar with additional context
with st.sidebar:
    st.header("Analysis Parameters")
    st.write("Total Budget: $5 Billion")
    st.write("Timeframe: 2010-2027")
    st.write("Base Year: 2010")
    
    st.markdown("---")
    
    st.header("Parameter Effects")
    st.write("Incentive Scaling: 20% per $1B")
    st.write("Growth Scaling: 15% per $1B")
    
    st.markdown("---")
    
    st.header("Data Sources")
    st.write("- Historical BEV adoption data (2010-2023)")
    st.write("- CA incentive program data")
    st.write("- Market growth parameters")
    
    st.markdown("---")
    
    st.header("Download Results")
    if st.button("Download Projection Results"):
        results_df = pd.DataFrame({
            'Year': projection_results['years'],
            'BEV_Vehicles': projection_results['bev']
        })
        results_df.to_excel('bev_projection_2010_2027.xlsx', index=False)
        st.success("Projection results downloaded successfully")
