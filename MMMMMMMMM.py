import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Set page config
st.set_page_config(layout="wide", page_title="BEV Adoption Analysis with RMSE")

# Load data
@st.cache_data
def load_data():
    data = pd.read_excel('ZEVdata.xlsx')
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

    def calculate_historical_rmse():
        """Calculate RMSE using all historical data"""
        # Initial conditions from 2010
        X0_2010 = [
            V_ICE_data[0]/V_ICE_mean,
            V_BEV_data[0]/V_BEV_mean,
            M_data[0]/M_mean,
            C_data[0]/C_mean,
            S_data[0]/S_mean
        ]
        
        # Solve for entire historical period
        years_span = len(years)
        t_eval = np.linspace(0, years_span-1, years_span)
        solution = solve_ivp(
            system,
            (0, years_span-1),
            X0_2010,
            args=(params,),
            t_eval=t_eval,
            method='RK45'
        )
        
        # Get predictions and calculate metrics
        predicted_bev = solution.y[1] * V_BEV_mean
        rmse = np.sqrt(np.mean((V_BEV_data - predicted_bev)**2))
        rmse_percentage = rmse / np.mean(V_BEV_data)
        mae = np.mean(np.abs(V_BEV_data - predicted_bev))
        mape = np.mean(np.abs((V_BEV_data - predicted_bev) / V_BEV_data)) * 100
        
        return {
            'rmse_absolute': rmse,
            'rmse_percentage': rmse_percentage,
            'mae': mae,
            'mape': mape,
            'years': years,
            'actual': V_BEV_data,
            'predicted': predicted_bev
        }

    def run_scenario_with_error(base_params, growth_ratio, incentive_ratio, rmse):
        """Run scenario with Monte Carlo error simulation starting from 2010"""
        scenario_params = base_params.copy()
        
        # Calculate budget effects
        growth_scaling = 1 + (growth_ratio * 5e9/4e9) * 0.15
        incentive_scaling = 1 + (incentive_ratio * 5e9/4e9) * 0.20
        
        # Modify parameters for scenario
        scenario_params['r2'] *= growth_scaling
        scenario_params['k_C'] *= incentive_scaling
        scenario_params['k_V'] *= incentive_scaling
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
        t_eval_historical = np.linspace(0, 14, 14*12+1)
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
        t_eval_future = np.linspace(0, 4, 4*12+1)
        solution_future = solve_ivp(
            system,
            (0, 4),
            X0_2024,
            args=(scenario_params,),
            t_eval=t_eval_future,
            method='RK45'
        )
        
        # Process results
        bev_historical = solution_historical.y[1] * V_BEV_mean
        bev_future = solution_future.y[1] * V_BEV_mean
        years_historical = np.array([2010 + t for t in solution_historical.t])
        years_future = np.array([2024 + t for t in solution_future.t])
        
        # Generate error simulations for future period
        n_simulations = 1000
        simulations = np.zeros((len(years_future), n_simulations))
        for i in range(n_simulations):
            errors = np.random.normal(0, rmse, size=len(years_future))
            simulations[:, i] = bev_future * (1 + errors)
        
        # Calculate confidence intervals
        lower_bound = np.percentile(simulations, 5, axis=1)
        upper_bound = np.percentile(simulations, 95, axis=1)
        mean_prediction = np.mean(simulations, axis=1)
        
        return {
            'historical_years': years_historical,
            'historical_bev': bev_historical,
            'future_years': years_future,
            'future_mean': mean_prediction,
            'future_lower': lower_bound,
            'future_upper': upper_bound,
            'future_simulations': simulations
        }

    # Calculate historical RMSE
    rmse_results = calculate_historical_rmse()

    # Create main title
    st.title("BEV Adoption Scenario Analysis with RMSE")
    st.write(f"Model Historical RMSE: {rmse_results['rmse_percentage']*100:.2f}%")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "RMSE Analysis",
        "Scenario Analysis",
        "Error Impact",
        "Monte Carlo Results"
    ])

    with tab1:
        st.header("Model RMSE Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Error Metrics")
            st.metric("RMSE (%)", f"{rmse_results['rmse_percentage']*100:.2f}%")
            st.metric("MAPE", f"{rmse_results['mape']:.2f}%")
            st.metric("MAE", f"{rmse_results['mae']/1e6:.2f}M vehicles")
        
        with col2:
            st.subheader("Historical Fit")
            
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Scatter(
                x=rmse_results['years'],
                y=rmse_results['actual']/1e6,
                name='Actual',
                mode='lines+markers',
                line=dict(color='blue')
            ))
            
            fig_hist.add_trace(go.Scatter(
                x=rmse_results['years'],
                y=rmse_results['predicted']/1e6,
                name='Predicted',
                mode='lines+markers',
                line=dict(color='red', dash='dash')
            ))
            
            fig_hist.update_layout(
                title='Model Validation (2010-2023)',
                xaxis_title='Year',
                yaxis_title='BEV Vehicles (Millions)',
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.subheader("Detailed Comparison")
        comparison_df = pd.DataFrame({
            'Year': rmse_results['years'],
            'Actual (M)': rmse_results['actual']/1e6,
            'Predicted (M)': rmse_results['predicted']/1e6,
            'Error (M)': (rmse_results['actual'] - rmse_results['predicted'])/1e6,
            'Error (%)': ((rmse_results['actual'] - rmse_results['predicted'])/rmse_results['actual'])*100
        })
        st.table(comparison_df.round(3))

    # Define scenarios
    scenarios = [
        ("Balanced", 0.5, 0.5),
        ("Growth Focus", 0.7, 0.3),
        ("Incentive Focus", 0.3, 0.7),
        ("High Growth", 0.8, 0.2),
        ("High Incentive", 0.2, 0.8)
    ]

    with tab2:
        st.header("Scenario Comparison with RMSE")
        
        fig = go.Figure()
        scenario_results = {}

        for name, growth_ratio, incentive_ratio in scenarios:
            results = run_scenario_with_error(
                params, growth_ratio, incentive_ratio, rmse_results['rmse_percentage']
            )
            
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
                y=results['future_mean']/1e6,
                name=f"{name} (Projected)",
                mode='lines',
                line=dict(width=2)
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=results['future_years'],
                y=results['future_upper']/1e6,
                name=f"{name} CI",
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=results['future_years'],
                y=results['future_lower']/1e6,
                name=f"{name} CI",
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,80,0.2)',
                showlegend=False
            ))

        # Add vertical line at 2024
        fig.add_vline(x=2024, line_dash="dash", line_color="gray", 
                     annotation_text="Projection Start")

        fig.update_layout(
            title=f'BEV Adoption Scenarios with Historical RMSE ({rmse_results["rmse_percentage"]*100:.1f}%)',
            xaxis_title='Year',
            yaxis_title='BEV Vehicles (Millions)',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Error Impact Analysis")
        
        impact_data = []
        for name in scenario_results:
            result = scenario_results[name]
            final_mean = result['future_mean'][-1]/1e6
            final_lower = result['future_lower'][-1]/1e6
            final_upper = result['future_upper'][-1]/1e6
            uncertainty = final_upper - final_lower
            
            impact_data.append({
                'Scenario': name,
                'Mean 2027 Adoption (M)': f"{final_mean:.2f}",
                'Lower 90% CI (M)': f"{final_lower:.2f}",
                'Upper 90% CI (M)': f"{final_upper:.2f}",
                'Uncertainty Range (M)': f"{uncertainty:.2f}"
            })
        
        st.table(pd.DataFrame(impact_data))

    with tab4:
        st.header("Monte Carlo Distribution Analysis")
        
        selected_scenario = st.selectbox(
            "Select scenario for detailed analysis",
            list(scenario_results.keys())
        )
        
        result = scenario_results[selected_scenario]
        
        fig_hist = plt.figure(figsize=(10, 6))
        plt.hist(
            result['future_simulations'][-1]/1e6,
            bins=50,
            density=True,
            alpha=0.7,
            color='skyblue',
            edgecolor='black'
        )
        plt.axvline(
            result['future_mean'][-1]/1e6,
            color='red',
            linestyle='--',
            label='Mean'
        )
        plt.axvline(
            result['future_lower'][-1]/1e6,
            color='green',
            linestyle=':',
            label='90% CI'
        )
        plt.axvline(
            result['future_upper'][-1]/1e6,
            color='green',
            linestyle=':',
            label='90% CI'
        )
        plt.title(f"Distribution of 2027 BEV Adoption - {selected_scenario}")
        plt.xlabel("BEV Vehicles (Millions)")
        plt.ylabel("Density")
        plt.legend()
        st.pyplot(fig_hist)

    # Sidebar information
    with st.sidebar:
        st.header("Analysis Parameters")
        st.write("Total Budget: $5 Billion")
        st.write("Period: 2024-2027")
        st.write("Initial BEVs: 1.7M")
        
        st.markdown("---")
        
        st.write("Error Metrics:")
        st.write(f"- Historical RMSE: {rmse_results['rmse_percentage']*100:.2f}%")
        st.write(f"- Historical MAPE: {rmse_results['mape']:.2f}%")
        st.write("- Confidence Level: 90%")
        st.write("- Monte Carlo Runs: 1,000")
        
        if st.button("Download Results"):
            # Create detailed results DataFrame
            results_df = pd.DataFrame({
                'Year': result['future_years'],
                **{f"{name} (Mean)": scenario_results[name]['future_mean']/1e6 
                   for name in scenario_results},
                **{f"{name} (Lower CI)": scenario_results[name]['future_lower']/1e6 
                   for name in scenario_results},
                **{f"{name} (Upper CI)": scenario_results[name]['future_upper']/1e6 
                   for name in scenario_results}
            })
            results_df.to_excel('bev_scenarios_with_rmse.xlsx', index=False)
            st.success("Results downloaded successfully")

except Exception as e:
    st.error(f"Error in analysis: {str(e)}")
    st.write("Please ensure all data files are available and properly formatted.")