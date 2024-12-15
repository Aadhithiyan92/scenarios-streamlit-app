import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.set_page_config(layout="wide", page_title="Synchronized vs Unsynchronized BEV Adoption Analysis")

st.title("Impact of Parameter Synchronization on BEV Adoption (2024-2027)")
st.write("Analyzing the effects of coordinating growth and incentive parameters in California")

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
def synchronized_scaling(t, growth_ratio, incentive_ratio, phase_shift=0.5):
    """
    Create synchronized scaling factors that coordinate growth and incentives
    
    Parameters:
    t: time point
    growth_ratio: base growth budget ratio
    incentive_ratio: base incentive budget ratio
    phase_shift: timing difference between growth and incentives (in years)
    """
    # Create wave-like coordination between parameters
    growth_phase = np.sin(2 * np.pi * (t + phase_shift) / 4)  # 4-year cycle
    incentive_phase = np.sin(2 * np.pi * t / 4)
    
    # Base scaling
    base_growth = 1 + (growth_ratio * 5e9/4e9) * 0.15
    base_incentive = 1 + (incentive_ratio * 5e9/4e9) * 0.20
    
    # Adjust scaling based on phase
    growth_scaling = base_growth * (1 + 0.2 * growth_phase)
    incentive_scaling = base_incentive * (1 + 0.2 * incentive_phase)
    
    return growth_scaling, incentive_scaling

def run_synchronized_scenario(base_params, growth_ratio, incentive_ratio, synchronized=True, years=4):
    """Run scenario with option for parameter synchronization"""
    scenario_params = base_params.copy()
    
    # Initial conditions for 2024 (1.7M BEVs)
    X0_2024 = [V_ICE_norm[0], 1.7e6/V_BEV_mean, M_norm[0], C_norm[0], S_norm[0]]
    
    if synchronized:
        # Create time points for evaluation
        t_eval = np.linspace(0, years, years*12+1)
        
        # Get synchronized scaling factors
        growth_scales, incentive_scales = zip(*[
            synchronized_scaling(t, growth_ratio, incentive_ratio)
            for t in t_eval
        ])
        
        # Apply synchronized scaling to parameters
        scenario_params['r2'] *= np.mean(growth_scales)
        scenario_params['k_C'] *= np.mean(incentive_scales)
        scenario_params['k_V'] *= np.mean(incentive_scales)
        scenario_params['beta1'] *= np.mean(incentive_scales)
    else:
        # Traditional unsynchronized scaling
        growth_scaling = 1 + (growth_ratio * 5e9/4e9) * 0.15
        incentive_scaling = 1 + (incentive_ratio * 5e9/4e9) * 0.20
        
        scenario_params['r2'] *= growth_scaling
        scenario_params['k_C'] *= incentive_scaling
        scenario_params['k_V'] *= incentive_scaling
        scenario_params['beta1'] *= incentive_scaling
    
    # Solve system
    t_eval = np.linspace(0, years, years*12+1)
    solution = solve_ivp(
        system,
        (0, years),
        X0_2024,
        args=(scenario_params,),
        t_eval=t_eval,
        method='RK45'
    )
    
    bev = solution.y[1] * V_BEV_mean
    years_array = np.array([2024 + t for t in solution.t])
    
    return years_array, bev

# Define scenarios
scenarios = [
    ("Balanced (Synchronized)", 0.5, 0.5, True),
    ("Balanced (Unsynchronized)", 0.5, 0.5, False),
    ("Growth Focus (Synchronized)", 0.7, 0.3, True),
    ("Growth Focus (Unsynchronized)", 0.7, 0.3, False),
    ("Incentive Focus (Synchronized)", 0.3, 0.7, True),
    ("Incentive Focus (Unsynchronized)", 0.3, 0.7, False)
]

# Create tabs
tab1, tab2, tab3 = st.tabs(["Comparative Analysis", "Impact Assessment", "Recommendations"])

with tab1:
    st.header("Synchronized vs Unsynchronized Approaches")
    
    # Run scenarios and create visualization
    fig = go.Figure()
    scenario_results = {}
    
    for name, growth_ratio, incentive_ratio, is_sync in scenarios:
        years, bev = run_synchronized_scenario(params, growth_ratio, incentive_ratio, is_sync)
        scenario_results[name] = (years, bev)
        
        line_style = 'solid' if is_sync else 'dash'
        fig.add_trace(go.Scatter(
            x=years,
            y=bev/1e6,
            name=name,
            line=dict(dash=line_style)
        ))
    
    fig.update_layout(
        title='BEV Adoption: Synchronized vs Unsynchronized Parameters',
        xaxis_title='Year',
        yaxis_title='BEV Vehicles (Millions)',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison metrics
    st.subheader("Impact of Synchronization")
    for base_type in ["Balanced", "Growth Focus", "Incentive Focus"]:
        sync_name = f"{base_type} (Synchronized)"
        unsync_name = f"{base_type} (Unsynchronized)"
        
        sync_final = scenario_results[sync_name][1][-1]
        unsync_final = scenario_results[unsync_name][1][-1]
        
        improvement = ((sync_final - unsync_final) / unsync_final * 100)
        
        st.metric(
            f"{base_type} Strategy",
            f"{sync_final/1e6:.2f}M vs {unsync_final/1e6:.2f}M",
            f"{improvement:+.1f}% with synchronization"
        )

with tab2:
    st.header("Impact Assessment")
    
    selected_scenario = st.selectbox(
        "Select scenario for detailed analysis:",
        [name for name, _, _, _ in scenarios]
    )
    
    years, bev = scenario_results[selected_scenario]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Adoption Metrics")
        final_adoption = bev[-1]
        improvement = (final_adoption - 1.7e6) / 1.7e6 * 100
        annual_growth = (np.power(final_adoption/1.7e6, 1/4) - 1) * 100
        
        st.metric("Final Adoption (2027)", f"{final_adoption/1e6:.2f}M vehicles")
        st.metric("Total Growth", f"{improvement:.1f}%")
        st.metric("Annual Growth Rate", f"{annual_growth:.1f}%")
    
    with col2:
        st.subheader("Parameter Effectiveness")
        # Calculate parameter effectiveness
        effectiveness = bev[-1] / (1.7e6 * (1 + improvement/100))
        st.metric("Parameter Efficiency", f"{effectiveness:.2f}")
        st.write("(Higher value indicates better parameter utilization)")

with tab3:
    st.header("Strategic Recommendations")
    
    # Find best performing scenario
    best_scenario = max(scenario_results.items(), key=lambda x: x[1][1][-1])
    
    st.markdown(f"""
    ### Key Findings
    
    1. **Optimal Strategy**: {best_scenario[0]}
       - Achieves highest adoption rate
       - Most efficient parameter utilization
    
    2. **Synchronization Benefits**:
       - Enhanced infrastructure-incentive coordination
       - Improved market response
       - More efficient resource utilization
    
    3. **Implementation Guidelines**:
       - Start with infrastructure development
       - Phase in incentives as infrastructure becomes available
       - Monitor and adjust synchronization timing
       
    4. **Regional Considerations**:
       - Urban areas: Focus on rapid infrastructure deployment
       - Rural areas: Longer infrastructure development phase
       - High-density areas: Accelerated incentive deployment
    """)

# Sidebar
with st.sidebar:
    st.header("Analysis Parameters")
    st.write("Budget: $5 Billion")
    st.write("Period: 2024-2027")
    st.write("Initial: 1.7M BEVs")
    
    st.markdown("---")
    
    st.write("Synchronization Effects:")
    st.write("- Growth Phase Shift: 0.5 years")
    st.write("- Coordination Window: 4 years")
    
    if st.button("Download Results"):
        results_df = pd.DataFrame({
            'Year': years,
            **{name: bev/1e6 for name, (years, bev) in scenario_results.items()}
        })
        results_df.to_excel('synchronized_scenario_results.xlsx', index=False)
        st.success("Results downloaded successfully")