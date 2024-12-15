import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="BEV Adoption Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #eee;
        padding-bottom: 0.5rem;
    }
    .stSelectbox {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
data = pd.read_excel("ZEVdata.xlsx")
years = data['Year'].values
V_ICE_data = data['V'].values
V_BEV_data = data['BEV'].values
V_PHEV_data = data['PHEV'].values
V_FCEV_data = data['FCEV'].values
M_data = data['M'].values
C_data = data['C'].values
S_data = data['S'].values

# Normalize data
V_ICE_mean, V_BEV_mean = np.mean(V_ICE_data), np.mean(V_BEV_data)
M_mean, C_mean, S_mean = np.mean(M_data), np.mean(C_data), np.mean(S_data)

V_ICE_norm = V_ICE_data / V_ICE_mean
V_BEV_norm = V_BEV_data / V_BEV_mean
M_norm = M_data / M_mean
C_norm = C_data / C_mean
S_norm = S_data / S_mean

# Base year and initial conditions
base_year = 2010
X0 = [V_ICE_norm[0], V_BEV_norm[0], M_norm[0], C_norm[0], S_norm[0]]

# Model parameters
params = {
    'r1': 0.04600309537728457,
    'K1': 19.988853430783674,
    'alpha1': 2.1990994330017567e-20,
    'r2': 0.4272847858514477,
    'beta1': 0.0009,
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

def incentive_time_effect(t, start_year, end_year, incentive_amount, decay_rate):
    if start_year <= t <= end_year:
        return incentive_amount * np.exp(-decay_rate * (t - start_year))
    return 0

def system(t, X, params):
    V, B, M, C, S = X
    
    # Unpack parameters
    r1, K1, alpha1, r2, beta1, gamma1, phi1, phi2, eta, psi1, psi2, delta, epsilon, \
    zeta, k_C, k_V, k_A, k_D, lambda_C, lambda_V, lambda_A, lambda_D, kappa, \
    lambda_S, omega, tau, cvrp_end_year = params.values()

    # Calculate incentive effects
    total_incentive = calculate_total_incentives(t, params)
    total_vehicles = V + B
    ev_fraction = B / total_vehicles if total_vehicles > 0 else 0

    # System equations for BEV focus
    dV_dt = r1 * V * (1 - total_vehicles/K1) * (1 - omega * ev_fraction) - tau * V * ev_fraction - epsilon * V
    dB_dt = r2 * B + beta1 * total_incentive + alpha1 * tau * V * ev_fraction - gamma1 * B
    dM_dt = phi1 * V + phi2 * B - eta * M
    dC_dt = (psi1 * V + psi2 * B) * M / total_vehicles - delta * C + zeta * (V / total_vehicles) ** 2
    dS_dt = kappa * B / total_vehicles - lambda_S * S

    return [dV_dt, dB_dt, dM_dt, dC_dt, dS_dt]

def calculate_total_incentives(t, params):
    year = t + base_year
    
    cvrp_effect = incentive_time_effect(year, 2010, params['cvrp_end_year'], 
                                      params['k_C'], params['lambda_C'])
    cvap_effect = incentive_time_effect(year, 2018, 2030, 
                                      params['k_V'], params['lambda_V'])
    cc4a_effect = incentive_time_effect(year, 2015, 2030, 
                                      params['k_A'], params['lambda_A'])
    dcap_effect = incentive_time_effect(year, 2016, 2030, 
                                      params['k_D'], params['lambda_D'])
    
    return cvrp_effect + cvap_effect + cc4a_effect + dcap_effect

def project_ev_adoption(projection_year, params):
    t_span = (0, projection_year - base_year)
    t_eval = np.arange(0, projection_year - base_year + 1, 1)

    solution = solve_ivp(system, t_span, X0, args=(params,), t_eval=t_eval, method='RK45')

    V_ICE_proj = solution.y[0] * V_ICE_mean
    V_BEV_proj = solution.y[1] * V_BEV_mean
    projection_years = np.arange(base_year, projection_year + 1)

    return projection_years, V_ICE_proj, V_BEV_proj

def create_sensitivity_params(param_type='growth'):
    base_params = params.copy()
    
    if param_type == 'growth':
        base_value = base_params['r2']
        param_key = 'r2'
    else:
        base_value = base_params['k_C']
        param_key = 'k_C'
    
    variations = {
        'Very Low (-20%)': 0.8,
        'Low (-10%)': 0.9,
        'Base': 1.0,
        'High (+10%)': 1.1,
        'Very High (+20%)': 1.2
    }
    
    sensitivity_params = {}
    for name, factor in variations.items():
        params_var = base_params.copy()
        params_var[param_key] = base_value * factor
        sensitivity_params[name] = params_var
    
    return sensitivity_params

def calculate_uncertainty_metrics(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    cv = (std_val / mean_val) * 100
    ci_lower = mean_val - 1.96 * std_val
    ci_upper = mean_val + 1.96 * std_val
    
    return {
        'mean': mean_val,
        'std': std_val,
        'cv': cv,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def main():
    st.title('🚗 BEV Adoption Analysis Dashboard')
    
    with st.sidebar:
        st.markdown("### Analysis Controls")
        st.markdown("---")
        
        analysis_type = st.selectbox(
            'Sensitivity Analysis Type',
            ['Growth Rate', 'Incentive Level']
        )
        
        projection_year = st.slider(
            'Projection Year',
            2023, 2030, 2027
        )
        
        cvrp_continues = st.checkbox(
            'Continue CVRP after 2023',
            True
        )
    
    try:
        # Generate scenarios
        sensitivity_params = create_sensitivity_params(
            'growth' if analysis_type == 'Growth Rate' else 'incentive'
        )
        
        scenarios = {}
        for name, params_set in sensitivity_params.items():
            params_set['cvrp_end_year'] = projection_year if cvrp_continues else 2023
            years, ice, bev = project_ev_adoption(projection_year, params_set)
            scenarios[name] = {'years': years, 'BEV': bev}
        
        # Create results dataframe
        results = []
        for scenario, data in scenarios.items():
            for year_idx, year in enumerate(data['years']):
                results.append({
                    'Scenario': scenario,
                    'Year': year,
                    'BEV_Count': data['BEV'][year_idx]
                })
        
        results_df = pd.DataFrame(results)
        
        # Display metrics
        st.markdown("### Key Metrics and Uncertainty Analysis")
        
        final_year = results_df[results_df['Year'] == projection_year]
        metrics = calculate_uncertainty_metrics(final_year['BEV_Count'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mean BEV Projection",
                f"{metrics['mean']:,.0f}",
                f"±{metrics['cv']:.1f}% uncertainty"
            )
        
        with col2:
            st.metric(
                "95% Confidence Interval",
                f"{metrics['ci_lower']:,.0f} - {metrics['ci_upper']:,.0f}"
            )
        
        with col3:
            st.metric(
                "Coefficient of Variation",
                f"{metrics['cv']:.1f}%"
            )
        
        # Interactive visualization
        st.markdown("### BEV Adoption Trajectories")
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        
        for i, (scenario, data) in enumerate(scenarios.items()):
            fig.add_trace(go.Scatter(
                x=data['years'],
                y=data['BEV'],
                name=scenario,
                line=dict(color=colors[i], width=3),
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title=f'BEV Adoption Under Different {analysis_type} Scenarios',
            xaxis_title="Year",
            yaxis_title="Number of BEVs",
            height=600,
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis tabs
        st.markdown("### 📊 Detailed Analysis")
        
        tab1, tab2 = st.tabs(["Scenario Comparison", "Data Table"])
        
        with tab1:
            final_year_data = results_df[results_df['Year'] == projection_year]
            
            comp_fig = go.Figure(data=[
                go.Bar(
                    x=final_year_data['Scenario'],
                    y=final_year_data['BEV_Count'],
                    text=final_year_data['BEV_Count'].apply(lambda x: f'{x:,.0f}'),
                    textposition='auto',
                )
            ])
            
            comp_fig.update_layout(
                title=f'BEV Adoption by Scenario in {projection_year}',
                xaxis_title="Scenario",
                yaxis_title="Number of BEVs",
                height=400
            )
            
            st.plotly_chart(comp_fig, use_container_width=True)
        
        with tab2:
            st.dataframe(
                results_df.pivot(
                    index='Year',
                    columns='Scenario',
                    values='BEV_Count'
                ).style.format("{:,.0f}")
            )
        
        # Download section
        st.markdown("### 💾 Download Results")
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Complete Analysis",
            csv,
            f'bev_analysis_{analysis_type}_{projection_year}.csv',
            'text/csv'
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.warning("Please check your input parameters and try again.")

if __name__ == "__main__":
    main()
