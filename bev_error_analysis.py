import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Set page config
st.set_page_config(layout="wide", page_title="BEV Adoption Scenario Analysis with Error Simulation")

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel('C:/Users/as3889/Desktop/ZEVdata.xlsx')
    return data

try:
    data = load_data()
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

    # Model parameters
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
        
        total_vehicles = V + B
        ev_fraction = B / total_vehicles if total_vehicles > 0 else 0

        dV_dt = params['r1'] * V * (1 - total_vehicles/params['K1']) * \
                (1 - params['omega'] * ev_fraction) - params['tau'] * V * ev_fraction - \
                params['epsilon'] * V
        dB_dt = params['r2'] * B + params['beta1'] * params['k_C'] + \
                params['alpha1'] * params['tau'] * V * ev_fraction - params['gamma1'] * B
        dM_dt = params['phi1'] * V + params['phi2'] * B - params['eta'] * M
        dC_dt = (params['psi1'] * V + params['psi2'] * B) * M / total_vehicles - \
                params['delta'] * C + params['zeta'] * (V / total_vehicles) ** 2
        dS_dt = params['kappa'] * B / total_vehicles - params['lambda_S'] * S

        return [dV_dt, dB_dt, dM_dt, dC_dt, dS_dt]

    def run_scenario_with_error(base_params, growth_ratio, incentive_ratio, rmse=0.1, n_simulations=1000):
        """Run scenario with Monte Carlo error simulation"""
        scenario_params = base_params.copy()
        
        # Calculate budget effects
        growth_scaling = 1 + (growth_ratio * 5e9/4e9) * 0.15
        incentive_scaling = 1 + (incentive_ratio * 5e9/4e9) * 0.20
        
        # Modify parameters
        scenario_params['r2'] *= growth_scaling
        scenario_params['k_C'] *= incentive_scaling
        scenario_params['k_V'] *= incentive_scaling
        scenario_params['beta1'] *= incentive_scaling
        
        # Initial conditions
        X0_2024 = [
            V_ICE_norm[0],
            1e6/V_BEV_mean,
            M_norm[0],
            C_norm[0],
            S_norm[0]
        ]
        
        # Solve base scenario
        t_eval = np.linspace(0, 4, 49)  # Monthly resolution
        solution = solve_ivp(
            system,
            (0, 4),
            X0_2024,
            args=(scenario_params,),
            t_eval=t_eval,
            method='RK45'
        )
        
        base_bev = solution.y[1] * V_BEV_mean
        years_array = np.array([2024 + t for t in solution.t])
        
        # Generate error simulations
        simulations = np.zeros((len(years_array), n_simulations))
        for i in range(n_simulations):
            errors = np.random.normal(0, rmse, size=len(years_array))
            simulations[:, i] = base_bev * (1 + errors)
        
        # Calculate confidence intervals
        lower_bound = np.percentile(simulations, 5, axis=1)
        upper_bound = np.percentile(simulations, 95, axis=1)
        mean_prediction = np.mean(simulations, axis=1)
        
        return years_array, mean_prediction, lower_bound, upper_bound, simulations

    # Create main title
    st.title("BEV Adoption Scenario Analysis with Error Simulation")
    st.write("Analyzing $5B Budget Allocation with Uncertainty (2024-2027)")

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Scenario Analysis",
        "Error Impact",
        "Monte Carlo Results"
    ])

    # Define scenarios
    scenarios = [
        ("Growth Focus", 0.7, 0.3),
        ("Incentive Focus", 0.3, 0.7)
    ]

    # Add RMSE control
    rmse = st.sidebar.slider(
        "Model RMSE (%)",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    ) / 100

    with tab1:
        st.header("Scenario Comparison with Uncertainty")
        
        fig = go.Figure()
        scenario_results = {}

        for name, growth_ratio, incentive_ratio in scenarios:
            years, mean_pred, lower, upper, sims = run_scenario_with_error(
                params, growth_ratio, incentive_ratio, rmse
            )
            
            scenario_results[name] = {
                'years': years,
                'mean': mean_pred,
                'lower': lower,
                'upper': upper,
                'simulations': sims
            }
            
            # Plot mean prediction
            fig.add_trace(go.Scatter(
                x=years,
                y=mean_pred/1e6,
                name=f"{name} (Mean)",
                mode='lines',
                line=dict(width=2)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=years,
                y=upper/1e6,
                name=f"{name} CI",
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=years,
                y=lower/1e6,
                name=f"{name} CI",
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                showlegend=False
            ))

        fig.update_layout(
            title='BEV Adoption Scenarios with 90% Confidence Intervals',
            xaxis_title='Year',
            yaxis_title='BEV Vehicles (Millions)',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Error Impact Analysis")
        
        # Calculate error impact metrics
        impact_data = []
        for name in scenario_results:
            result = scenario_results[name]
            final_mean = result['mean'][-1]/1e6
            final_lower = result['lower'][-1]/1e6
            final_upper = result['upper'][-1]/1e6
            uncertainty = final_upper - final_lower
            
            impact_data.append({
                'Scenario': name,
                'Mean 2027 Adoption (M)': f"{final_mean:.2f}",
                'Lower 95% CI (M)': f"{final_lower:.2f}",
                'Upper 95% CI (M)': f"{final_upper:.2f}",
                'Uncertainty Range (M)': f"{uncertainty:.2f}"
            })
        
        st.table(pd.DataFrame(impact_data))

    with tab3:
        st.header("Monte Carlo Simulation Results")
        
        selected_scenario = st.selectbox(
            "Select scenario for detailed analysis",
            list(scenario_results.keys())
        )
        
        result = scenario_results[selected_scenario]
        
        # Plot histogram of final year results
        fig_hist = plt.figure(figsize=(10, 6))
        plt.hist(result['simulations'][-1]/1e6, bins=50, density=True)
        plt.title(f"Distribution of 2027 BEV Adoption - {selected_scenario}")
        plt.xlabel("BEV Vehicles (Millions)")
        plt.ylabel("Density")
        st.pyplot(fig_hist)

    # Sidebar information
    with st.sidebar:
        st.header("Analysis Parameters")
        st.write("Total Budget: $5 Billion")
        st.write("Period: 2024-2027")
        st.write("Initial BEVs: 1M")
        
        st.markdown("---")
        
        st.write("Simulation Settings:")
        st.write(f"- RMSE: {rmse*100:.1f}%")
        st.write("- Confidence Level: 90%")
        st.write("- Monte Carlo Runs: 1000")
        
        if st.button("Download Results"):
            results_df = pd.DataFrame({
                'Year': years,
                **{f"{name} (Mean)": result['mean']/1e6 for name, result in scenario_results.items()},
                **{f"{name} (Lower)": result['lower']/1e6 for name, result in scenario_results.items()},
                **{f"{name} (Upper)": result['upper']/1e6 for name, result in scenario_results.items()}
            })
            results_df.to_excel('bev_scenario_results_with_error.xlsx', index=False)
            st.success("Results downloaded successfully")

except Exception as e:
    st.error(f"Error in analysis: {str(e)}")
    st.write("Please ensure all data files are available and properly formatted.")