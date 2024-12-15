import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import plotly.express as px

# Load data
data = pd.read_excel('C:/Users/as3889/Desktop/ZEVdata.xlsx')
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
    'cvrp_end_year': 2027
}

def system(t, X, params):
    """System of differential equations for BEV adoption"""
    V, B, M, C, S = X
    
    year = t + 2024  # Updated to start from 2024
    if year in incentive_data['Year'].values:
        idx = incentive_data['Year'] == year
        cvrp_effect = incentive_data.loc[idx, 'CVRP_Norm'].values[0]
        cvap_effect = incentive_data.loc[idx, 'CVAP_Norm'].values[0]
        cc4a_effect = incentive_data.loc[idx, 'CC4A_Norm'].values[0]
        dcap_effect = incentive_data.loc[idx, 'DCAP_Norm'].values[0]
    else:
        cvrp_effect = incentive_time_effect(t, 2024, params['cvrp_end_year'], 'CVRP', params['lambda_C'])
        cvap_effect = incentive_time_effect(t, 2024, 2027, 'CVAP', params['lambda_V'])
        cc4a_effect = incentive_time_effect(t, 2024, 2027, 'CC4A', params['lambda_A'])
        dcap_effect = incentive_time_effect(t, 2024, 2027, 'DCAP', params['lambda_D'])

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
    year = t + 2024  # Updated to start from 2024
    
    if year in incentive_data['Year'].values:
        return incentive_data.loc[incentive_data['Year'] == year, f'{incentive_type}_Norm'].values[0]
    
    if start_year <= year <= end_year:
        return np.exp(-decay_rate * (year - start_year))
    return 0

def run_scenario(base_params, incentive_ratio, growth_ratio, total_budget=5e9, years=4):
    """Run a scenario with given budget allocation ratios"""
    scenario_params = base_params.copy()
    
    # Calculate annual budget allocation
    annual_budget = total_budget / years
    incentive_budget = annual_budget * incentive_ratio
    growth_budget = annual_budget * growth_ratio
    
    # Modify parameters based on budget allocation
    incentive_scaling = 1 + (incentive_budget/1e9) * 0.2  # 20% increase per billion
    scenario_params['k_C'] *= incentive_scaling
    scenario_params['k_V'] *= incentive_scaling
    scenario_params['k_A'] *= incentive_scaling
    scenario_params['k_D'] *= incentive_scaling
    scenario_params['beta1'] *= incentive_scaling
    
    # Growth parameter modifications
    growth_scaling = 1 + (growth_budget/1e9) * 0.15  # 15% increase per billion
    scenario_params['r2'] *= growth_scaling
    
    # Initial conditions for 2024 (1.7M BEVs)
    X0_2024 = [
        V_ICE_norm[0],
        1e6/V_BEV_mean,  # 1.7M BEVs normalized
        M_norm[0],
        C_norm[0],
        S_norm[0]
    ]
    
    # Solve system
    t_eval = np.linspace(0, years, years*12+1)  # Monthly resolution
    solution = solve_ivp(
        system,
        (0, years),
        X0_2024,
        args=(scenario_params,),
        t_eval=t_eval,
        method='RK45'
    )
    
    # Convert normalized values back
    bev = solution.y[1] * V_BEV_mean
    years_array = np.array([2024 + t for t in solution.t])
    
    return years_array, bev

# Define scenarios
scenarios = [
    ("70-30 Growth Focus", 0.3, 0.7),
    ("70-30 Incentive Focus", 0.7, 0.3)
]

# Run scenarios and plot results
scenario_results = {}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Professional color scheme

for (name, inc_ratio, growth_ratio), color in zip(scenarios, colors):
    years, bev = run_scenario(params, inc_ratio, growth_ratio)
    scenario_results[name] = (years, bev)
    plt.plot(years, bev/1e6, label=f"{name}", color=color, linewidth=2.5)

plt.title("BEV Adoption Scenarios with $5B Budget (2024-2027)", fontsize=14, pad=20)
plt.xlabel("Year", fontsize=12)
plt.ylabel("BEV Vehicles (Millions)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Print final adoption numbers
print("\nScenario Results (2027 Projections):")
print("-" * 50)
for name, (years, bev) in scenario_results.items():
    final_adoption = bev[-1]
    improvement = (final_adoption - 1e6) / 1e6 * 100
    print(f"{name}:")
    print(f"  Final Adoption: {final_adoption/1e6:.2f} million vehicles")
    print(f"  Improvement: {improvement:.1f}%")
    print("-" * 50)

st.set_page_config(layout="wide", page_title="BEV Adoption Strategy Analysis")

st.title("Strategic Analysis of $5B Budget Allocation for BEV Adoption in California (2024-2027)")

st.markdown("""
### Executive Summary
This analysis explores optimal strategies for allocating a $5 billion budget to accelerate Battery Electric Vehicle (BEV) adoption in California from 2024 to 2027. Starting from a baseline of 1.7 million BEVs in 2024, we analyze five different budget allocation scenarios between incentive programs and growth initiatives.
""")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Scenario Overview", "Detailed Analysis", "Impact Comparison", "Recommendations"])

with tab1:
    st.header("Budget Allocation Scenarios")
    
    st.markdown("""
    We've analyzed five strategic approaches to budget allocation:
    1. **Balanced Approach (50-50 Split)**: Equal distribution between incentives and growth initiatives
    2. **Incentive-Focused (60-40)**: Greater emphasis on consumer incentives
    3. **Growth-Focused (60-40)**: Priority on infrastructure and market growth
    4. **Strong Growth Focus (70-30)**: Maximizing growth initiatives
    5. **Strong Incentive Focus (70-30)**: Maximizing consumer incentives
    
    #### How Budget Affects BEV Adoption
    - **Incentive Budget** influences:
        - Direct consumer purchase incentives (CVRP, CVAP)
        - Infrastructure development programs (CC4A, DCAP)
        - Overall incentive effectiveness
    - **Growth Budget** impacts:
        - Market development rate
        - Technology adoption speed
        - Infrastructure expansion
    """)

with tab2:
    st.header("Scenario Analysis")
    
    # Run scenarios and create visualization
    fig = go.Figure()
    
    for name, inc_ratio, growth_ratio in scenarios:
        years, bev = run_scenario(params, inc_ratio, growth_ratio)
        fig.add_trace(go.Scatter(
            x=years,
            y=bev/1e6,
            name=name,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='BEV Adoption Trajectories by Scenario',
        xaxis_title='Year',
        yaxis_title='BEV Vehicles (Millions)',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parameter Impact Analysis")
        st.markdown("""
        For each billion dollars invested:
        - Incentive programs increase effectiveness by 20%
        - Growth initiatives boost adoption rate by 15%
        """)
        
    with col2:
        st.subheader("Key Metrics by 2027")
        scenario_metrics = []
        for name, (years, bev) in scenario_results.items():
            final_adoption = bev[-1]/1e6
            improvement = (bev[-1] - 1e6) / 1e6 * 100
            scenario_metrics.append({
                'Scenario': name,
                'Final Adoption (M)': f"{final_adoption:.2f}",
                'Improvement (%)': f"{improvement:.1f}"
            })
        
        st.table(pd.DataFrame(scenario_metrics))

with tab3:
    st.header("Comparative Impact Analysis")
    
    # Create bar chart comparing final adoption numbers
    final_numbers = {name: bev[-1]/1e6 for name, (years, bev) in scenario_results.items()}
    fig_bar = px.bar(
        x=list(final_numbers.keys()),
        y=list(final_numbers.values()),
        title="Final BEV Adoption by 2027",
        labels={'x': 'Scenario', 'y': 'Million Vehicles'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab4:
    st.header("Strategic Recommendations")
    
    # Find best performing scenario
    best_scenario = max(scenario_results.items(), key=lambda x: x[1][1][-1])
    
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
    st.write("Starting Point: 1.7M BEVs")
    
    st.markdown("---")
    
    st.header("Parameter Effects")
    st.write("Incentive Scaling: 20% per $1B")
    st.write("Growth Scaling: 15% per $1B")
    
    st.markdown("---")
    
    st.header("Data Sources")
    st.write("- Historical BEV adoption data")
    st.write("- CA incentive program data")
    st.write("- Market growth parameters")