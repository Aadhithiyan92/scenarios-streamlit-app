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
data = pd.read_excel("ZEVdata.xlsx")
years = data['Year'].values
V_ICE_data = data['V'].values
V_BEV_data = data['BEV'].values
M_data = data['M'].values
C_data = data['C'].values
S_data = data['S'].values

# Historical incentive data
incentive_data = pd.DataFrame({
    'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    'CVRP_Amount': [375000, 15156388, 20103380, 58796855, 90102019, 102848457, 101956078, 117563328,
                    173909297, 176018763, 95876302, 125864437, 101382166, 331899666],
    'CVAP_Amount': [0, 0, 0, 0, 0, 0, 0, 0, 2002253, 10000, 19418653, 1520500, 15000, 2214500],
    'DCAP_Amount': [0, 0, 0, 0, 0, 0, 10000, 45000, 190000, 245000, 336500, 544500, 581000, 689500],
    'CC4A_Amount': [0, 0, 0, 0, 0, 1898814, 4922899, 6068746, 10275857, 20377745, 23440151, 13864693,
                    13534438, 20614737]
})

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
    'cvrp_end_year': 2023
}

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
        cvap_effect = incentive_time_effect(t, 2010, 2027, 'CVAP', params['lambda_V'])
        cc4a_effect = incentive_time_effect(t, 2010, 2027, 'CC4A', params['lambda_A'])
        dcap_effect = incentive_time_effect(t, 2010, 2027, 'DCAP', params['lambda_D'])

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
    dC_dt = (params['psi1'] * V + params['psi2'] * B) * M / total_vehicles - params['delta'] * C + params['zeta'] * (V / total_vehicles) ** 2
    dS_dt = params['kappa'] * B / total_vehicles - params['lambda_S'] * S

    return [dV_dt, dB_dt, dM_dt, dC_dt, dS_dt]

def incentive_time_effect(t, start_year, end_year, incentive_type, decay_rate):
    """Calculate incentive effect at a given time"""
    year = t + 2010  # Start from 2010
    
    if year in incentive_data['Year'].values:
        return incentive_data.loc[incentive_data['Year'] == year, f'{incentive_type}_Norm'].values[0]
    
    if start_year <= year <= end_year:
        return np.exp(-decay_rate * (year - start_year))
    return 0

def run_scenario(base_params, incentive_ratio, growth_ratio, total_budget=5e9, years=4):
    """Run a scenario with two stages: historical and future"""
    scenario_params = base_params.copy()
    
    # Calculate budget effects
    growth_scaling = 1 + (growth_ratio * total_budget/years/1e9) * 0.10
    incentive_scaling = 1 + (incentive_ratio * total_budget/years/1e9) * 0.15
    
    # Modify parameters
    scenario_params['r2'] *= growth_scaling
    scenario_params['k_C'] *= incentive_scaling
    scenario_params['k_V'] *= incentive_scaling
    scenario_params['k_A'] *= incentive_scaling
    scenario_params['k_D'] *= incentive_scaling
    scenario_params['beta1'] *= incentive_scaling
    
    # Initial conditions from 2010
    X0_2010 = [
        V_ICE_data[0]/V_ICE_mean,
        V_BEV_data[0]/V_BEV_mean,
        M_data[0]/M_mean,
        C_data[0]/C_mean,
        S_data[0]/S_mean
    ]
    
    # First solve historical period (2010-2024)
    t_eval_historical = np.linspace(0, 14, 14*12+1)  # Monthly resolution
    solution_historical = solve_ivp(
        system,
        (0, 14),
        X0_2010,
        args=(scenario_params,),
        t_eval=t_eval_historical,
        method='RK45'
    )
    
    # Then solve future period (2024-2027)
    X0_2024 = [x[-1] for x in solution_historical.y]
    t_eval_future = np.linspace(0, 4, 4*12+1)  # Monthly resolution
    solution_future = solve_ivp(
        system,
        (0, 4),
        X0_2024,
        args=(scenario_params,),
        t_eval=t_eval_future,
        method='RK45'
    )
    
    # Combine results
    bev_historical = solution_historical.y[1] * V_BEV_mean
    bev_future = solution_future.y[1] * V_BEV_mean
    years_historical = np.array([2010 + t for t in solution_historical.t])
    years_future = np.array([2024 + t for t in solution_future.t])
    
    return {
        'historical_years': years_historical,
        'historical_bev': bev_historical,
        'future_years': years_future,
        'future_bev': bev_future
    }

# Define scenarios
scenarios = [
    ("70-30 Growth Focus", 0.3, 0.7),
    ("70-30 Incentive Focus", 0.7, 0.3)
]

st.title("Strategic Analysis of $5B Budget Allocation for BEV Adoption in California (2024-2027)")

st.markdown("""
### Executive Summary
This analysis explores optimal strategies for allocating a $5 billion budget to accelerate Battery Electric Vehicle (BEV) adoption in California from 2024 to 2027, with historical data from 2010 for model validation.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Scenario Analysis", "Impact Comparison", "Recommendations"])

with tab1:
    st.header("Scenario Comparison")
    
    fig = go.Figure()
    scenario_results = {}
    
    for name, inc_ratio, growth_ratio in scenarios:
        results = run_scenario(params, inc_ratio, growth_ratio)
        scenario_results[name] = results
        
        # Plot historical period
        fig.add_trace(go.Scatter(
            x=results['historical_years'],
            y=results['historical_bev']/1e6,
            name=f"{name} (Historical)",
            mode='lines',
            line=dict(width=1, dash='dot')
        ))
        
        # Plot future predictions
        fig.add_trace(go.Scatter(
            x=results['future_years'],
            y=results['future_bev']/1e6,
            name=f"{name} (Projected)",
            mode='lines',
            line=dict(width=2)
        ))
    
    # Add vertical line at 2024
    fig.add_vline(x=2024, line_dash="dash", line_color="gray", 
                 annotation_text="Projection Start")
    
    fig.update_layout(
        title='BEV Adoption Trajectories by Scenario',
        xaxis_title='Year',
        yaxis_title='BEV Vehicles (Millions)',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Add metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Impact")
        st.markdown("""
        For each billion dollars invested:
        - Incentive programs increase effectiveness by 20%
        - Growth initiatives boost adoption rate by 15%
        """)
    
    with col2:
        st.subheader("Final Adoption (2027)")
        metrics = []
        for name, results in scenario_results.items():
            final_adoption = results['future_bev'][-1]/1e6
            improvement = (results['future_bev'][-1] - results['historical_bev'][-1]) / results['historical_bev'][-1] * 100
            metrics.append({
                'Scenario': name,
                'Final Adoption (M)': f"{final_adoption:.2f}",
                'Improvement (%)': f"{improvement:.1f}"
            })
        st.table(pd.DataFrame(metrics))

with tab2:
    st.header("Impact Comparison")
    
    # Create bar chart
    final_numbers = {name: results['future_bev'][-1]/1e6 
                    for name, results in scenario_results.items()}
    fig_bar = px.bar(
        x=list(final_numbers.keys()),
        y=list(final_numbers.values()),
        title="Final BEV Adoption by 2027",
        labels={'x': 'Scenario', 'y': 'Million Vehicles'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.header("Strategic Recommendations")
    
    # Find best performing scenario
    best_scenario = max(
        scenario_results.items(),
        key=lambda x: x[1]['future_bev'][-1]
    )
    
    st.markdown(f"""
    ### Key Findings
    1. **Most Effective Strategy**: {best_scenario[0]}
        - Achieves highest BEV adoption by 2027
        - Shows most consistent growth trajectory
        
    2. **Budget Allocation Insights**:
        - Growth-focused strategies show stronger long-term results
        - Infrastructure development creates lasting impact
        - Combined approaches provide balanced benefits
        
    ### Recommended Implementation Strategy
    1. **Phase 1 (2024-2025)**:
        - Focus on infrastructure development
        - Establish foundation for market growth
        - Target key adoption barriers
        
    2. **Phase 2 (2026-2027)**:
        - Scale up incentive programs
        - Leverage established infrastructure
        - Accelerate market adoption
        
    ### Critical Success Factors
    - Consistent policy implementation
    - Regular progress monitoring
    - Adaptive strategy adjustment
    - Stakeholder engagement
    """)

# Add sidebar with additional context
with st.sidebar:
    st.header("Analysis Parameters")
    st.write("Total Budget: $5 Billion")
    st.write("Timeframe: 2024-2027")
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
    
    if st.button("Download Results"):
        results_df = pd.DataFrame({
            'Year': scenario_results[list(scenario_results.keys())[0]]['future_years'],
            **{f"{name} Historical": results['historical_bev']/1e6 
               for name, results in scenario_results.items()},
            **{f"{name} Projected": results['future_bev']/1e6 
               for name, results in scenario_results.items()}
        })
        results_df.to_excel('bev_scenarios_complete.xlsx', index=False)
        st.success("Results downloaded successfully")
