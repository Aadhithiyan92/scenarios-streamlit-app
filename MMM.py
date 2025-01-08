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
    data = pd.read_excel("ZEVdata.xlsx")
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

try:
    data, incentive_data = load_data()
    years = data['Year'].values
    V_ICE_data = data['V'].values
    V_BEV_data = data['BEV'].values
    V_PHEV_data = data['PHEV'].values
    V_FCEV_data = data['FCEV'].values
    M_data = data['M'].values
    C_data = data['C'].values
    S_data = data['S'].values

    # Calculate incentive normalizations
    total_incentives = 1511901636 + 25180906 + 121114580 + 2685500
    total_vehicles = (426921 + 152330 + 14305) + (3749 + 1121 + 68) + (3597 + 9648 + 144) + (221 + 279 + 7)
    mean_incentive = total_incentives / total_vehicles

    cvrp_incentive_norm = (1511901636 / (426921 + 152330 + 14305)) / mean_incentive
    cvap_incentive_norm = (25180906 / (3749 + 1121 + 68)) / mean_incentive
    cc4a_incentive_norm = (121114580 / (3597 + 9648 + 144)) / mean_incentive
    dcap_incentive_norm = (2685500 / (221 + 279 + 7)) / mean_incentive

    # Normalize data
    V_ICE_mean = np.mean(V_ICE_data)
    V_BEV_mean = np.mean(V_BEV_data)
    V_PHEV_mean = np.mean(V_PHEV_data)
    V_FCEV_mean = np.mean(V_FCEV_data)
    M_mean = np.mean(M_data)
    C_mean = np.mean(C_data)
    S_mean = np.mean(S_data)

    V_ICE_norm = V_ICE_data / V_ICE_mean
    V_BEV_norm = V_BEV_data / V_BEV_mean
    V_PHEV_norm = V_PHEV_data / V_PHEV_mean
    V_FCEV_norm = V_FCEV_data / V_FCEV_mean
    M_norm = M_data / M_mean
    C_norm = C_data / C_mean
    S_norm = S_data / S_mean

    # Model parameters
    params = {
        'r1': 0.04600309537728457,
        'K1': 19.988853430783674,
        'alpha1': 2.1990994330017567e-20,
        'alpha2': 7.217511478131137,
        'alpha3': 8.363375552715492,
        'r2': 0.4242847858514477,
        'beta1': 0.0010,
        'gamma1': 0.006195813413796029,
        'r3': 0.11,
        'beta2': 0.0004,
        'gamma2': 0.07,
        'r4': 0.14351479815351475,
        'beta3': 0.0001,
        'gamma3': 0.08,
        'phi1': 2.420439212993679,
        'phi2': 0.03290625472523172,
        'phi3': 7.647749286214547e-33,
        'phi4': 2.830388633928076e-32,
        'eta': 2.472995227365455,
        'psi1': 0.877999000323586,
        'psi2': 0.8825501230310412,
        'psi3': 1.3630983704958457,
        'psi4': 6.441751857606793e-08,
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
        """Calculate the effect of an incentive at a given time"""
        if start_year <= t <= end_year:
            return incentive_amount * np.exp(-decay_rate * (t - start_year))
        else:
            return 0

    def system(t, X, params):
        """System of differential equations for the ZEV adoption model"""
        V, B, P, F, M, C, S = X
        
        year = t + 2010  # Base year
        
        # Get incentive effects
        cvrp_effect = incentive_time_effect(year, 2010, params['cvrp_end_year'], 
                                          params['k_C'] * cvrp_incentive_norm, params['lambda_C'])
        cvap_effect = incentive_time_effect(year, 2018, 2030, 
                                          params['k_V'] * cvap_incentive_norm, params['lambda_V'])
        cc4a_effect = incentive_time_effect(year, 2015, 2030, 
                                          params['k_A'] * cc4a_incentive_norm, params['lambda_A'])
        dcap_effect = incentive_time_effect(year, 2016, 2030, 
                                          params['k_D'] * dcap_incentive_norm, params['lambda_D'])
        
        total_incentive = cvrp_effect + cvap_effect + cc4a_effect + dcap_effect
        total_vehicles = V + B + P + F
        ev_fraction = (B + P + F) / total_vehicles if total_vehicles > 0 else 0

        # System equations
        dV_dt = (params['r1'] * V * (1 - total_vehicles/params['K1']) * 
                 (1 - params['omega'] * ev_fraction) - params['tau'] * V * ev_fraction - 
                 params['epsilon'] * V)
        
        dB_dt = (params['r2'] * B + params['beta1'] * total_incentive + 
                 params['alpha1'] * params['tau'] * V * ev_fraction - params['gamma1'] * B)
        
        dP_dt = (params['r3'] * P + params['beta2'] * total_incentive + 
                 params['alpha2'] * params['tau'] * V * ev_fraction - params['gamma2'] * P)
        
        dF_dt = (params['r4'] * F + params['beta3'] * total_incentive + 
                 params['alpha3'] * params['tau'] * V * ev_fraction - params['gamma3'] * F)
        
        dM_dt = (params['phi1'] * V + params['phi2'] * B + params['phi3'] * P + 
                 params['phi4'] * F - params['eta'] * M)
        
        dC_dt = ((params['psi1'] * V + params['psi2'] * B + params['psi3'] * P + 
                  params['psi4'] * F) * M / total_vehicles - params['delta'] * C + 
                 params['zeta'] * (V / total_vehicles) ** 2)
        
        dS_dt = (params['kappa'] * (B + P + F) / total_vehicles - params['lambda_S'] * S)

        return [dV_dt, dB_dt, dP_dt, dF_dt, dM_dt, dC_dt, dS_dt]

    def calculate_historical_rmse():
        """Calculate RMSE using all historical data"""
        # Initial conditions from 2010
        X0_2010 = [
            V_ICE_norm[0],
            V_BEV_norm[0],
            V_PHEV_norm[0],
            V_FCEV_norm[0],
            M_norm[0],
            C_norm[0],
            S_norm[0]
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
        predicted_phev = solution.y[2] * V_PHEV_mean
        predicted_fcev = solution.y[3] * V_FCEV_mean
        
        # Calculate total ZEV predictions and actuals
        predicted_zev = predicted_bev + predicted_phev + predicted_fcev
        actual_zev = V_BEV_data + V_PHEV_data + V_FCEV_data
        
        rmse = np.sqrt(np.mean((actual_zev - predicted_zev)**2))
        rmse_percentage = rmse / np.mean(actual_zev)
        mae = np.mean(np.abs(actual_zev - predicted_zev))
        mape = np.mean(np.abs((actual_zev - predicted_zev) / actual_zev)) * 100
        
        return {
            'rmse_absolute': rmse,
            'rmse_percentage': rmse_percentage,
            'mae': mae,
            'mape': mape,
            'years': years,
            'actual_zev': actual_zev,
            'predicted_zev': predicted_zev,
            'predicted_bev': predicted_bev,
            'predicted_phev': predicted_phev,
            'predicted_fcev': predicted_fcev
        }

    def run_scenario_with_error(base_params, growth_ratio, incentive_ratio, rmse):
        """Run scenario with Monte Carlo error simulation"""
        scenario_params = base_params.copy()
        
        # Calculate budget effects
        growth_scaling = 1 + (growth_ratio * 5e9/4e9) * 0.13
        incentive_scaling = 1 + (incentive_ratio * 5e9/4e9) * 0.18
        
        # Modify parameters for scenario
        scenario_params['r2'] *= growth_scaling
        scenario_params['r3'] *= growth_scaling
        scenario_params['r4'] *= growth_scaling
        scenario_params['k_C'] *= incentive_scaling
        scenario_params['k_V'] *= incentive_scaling
        scenario_params['k_A'] *= incentive_scaling
        scenario_params['k_D'] *= incentive_scaling
        scenario_params['beta1'] *= incentive_scaling
        scenario_params['beta2'] *= incentive_scaling
        scenario_params['beta3'] *= incentive_scaling
        
        # Initial conditions from 2010
        X0_2010 = [
            V_ICE_norm[0],
            V_BEV_norm[0],
            V_PHEV_norm[0],
            V_FCEV_norm[0],
            M_norm[0],
            C_norm[0],
            S_norm[0]
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
        years_historical = np.array([2010 + t for t in solution_historical.t])
        years_future = np.array([2024 + t for t in solution_future.t])
        
        # Historical values
        ice_historical = solution_historical.y[0] * V_ICE_mean
        bev_historical = solution_historical.y[1] * V_BEV_mean
        phev_historical = solution_historical.y[2] * V_PHEV_mean
        fcev_historical = solution_historical.y[3] * V_FCEV_mean
        
        # Future values
        ice_future = solution_future.y[0] * V_ICE_mean
        bev_future = solution_future.y[1] * V_BEV_mean
        phev_future = solution_future.y[2] * V_PHEV_mean
        fcev_future = solution_future.y[3] * V_FCEV_mean
        
        # Generate error simulations for future period
        n_simulations = 1000
        bev_simulations = np.zeros((len(years_future), n_simulations))
        phev_simulations = np.zeros((len(years_future), n_simulations))
        fcev_simulations = np.zeros((len(years_future), n_simulations))
        
        for i in range(n_simulations):
            bev_errors = np.random.normal(0, rmse, size=len(years_future))
            phev_errors = np.random.normal(0, rmse, size=len(years_future))
            fcev_errors = np.random.normal(0, rmse, size=len(years_future))
            
            bev_simulations[:, i] = bev_future * (1 + bev_errors)
            phev_simulations[:, i] = phev_future * (1 + phev_errors)
            fcev_simulations[:, i] = fcev_future * (1 + fcev_errors)
        
        # Calculate confidence intervals for each vehicle type
        bev_lower = np.percentile(bev_simulations, 5, axis=1)
        bev_upper = np.percentile(bev_simulations, 95, axis=1)
        bev_mean = np.mean(bev_simulations, axis=1)
        
        phev_lower = np.percentile(phev_simulations, 5, axis=1)
        phev_upper = np.percentile(phev_simulations, 95, axis=1)
        phev_mean = np.mean(phev_simulations, axis=1)
        
        fcev_lower = np.percentile(fcev_simulations, 5, axis=1)
        fcev_upper = np.percentile(fcev_simulations, 95, axis=1)
        fcev_mean = np.mean(fcev_simulations, axis=1)
        
        return {
            'historical_years': years_historical,
            'historical_ice': ice_historical,
            'historical_bev': bev_historical,
            'historical_phev': phev_historical,
            'historical_fcev': fcev_historical,
            'future_years': years_future,
            'future_ice': ice_future,
            'future_bev_mean': bev_mean,
            'future_bev_lower': bev_lower,
            'future_bev_upper': bev_upper,
            'future_phev_mean': phev_mean,
            'future_phev_lower': phev_lower,
            'future_phev_upper': phev_upper,
            'future_fcev_mean': fcev_mean,
            'future_fcev_lower': fcev_lower,
            'future_fcev_upper': fcev_upper,
            'bev_simulations': bev_simulations,
            'phev_simulations': phev_simulations,
            'fcev_simulations': fcev_simulations
        }

    # Calculate historical RMSE
    rmse_results = calculate_historical_rmse()

    # Create main title
    st.title("ZEV Adoption Scenario Analysis with RMSE")
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
                y=rmse_results['actual_zev']/1e6,
                name='Actual ZEVs',
                mode='lines+markers',
                line=dict(color='blue')
            ))
            
            fig_hist.add_trace(go.Scatter(
                x=rmse_results['years'],
                y=rmse_results['predicted_zev']/1e6,
                name='Predicted ZEVs',
                mode='lines+markers',
                line=dict(color='red', dash='dash')
            ))
            
            # Add individual vehicle type predictions
            fig_hist.add_trace(go.Scatter(
                x=rmse_results['years'],
                y=rmse_results['predicted_bev']/1e6,
                name='Predicted BEVs',
                mode='lines',
                line=dict(color='green', dash='dot')
            ))
            
            fig_hist.add_trace(go.Scatter(
                x=rmse_results['years'],
                y=rmse_results['predicted_phev']/1e6,
                name='Predicted PHEVs',
                mode='lines',
                line=dict(color='orange', dash='dot')
            ))
            
            fig_hist.add_trace(go.Scatter(
                x=rmse_results['years'],
                y=rmse_results['predicted_fcev']/1e6,
                name='Predicted FCEVs',
                mode='lines',
                line=dict(color='purple', dash='dot')
            ))
            
            fig_hist.update_layout(
                title='Model Validation (2010-2023)',
                xaxis_title='Year',
                yaxis_title='Vehicles (Millions)',
                height=400
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.subheader("Detailed Comparison")
        comparison_df = pd.DataFrame({
            'Year': rmse_results['years'],
            'Actual ZEVs (M)': rmse_results['actual_zev']/1e6,
            'Predicted ZEVs (M)': rmse_results['predicted_zev']/1e6,
            'Error (M)': (rmse_results['actual_zev'] - rmse_results['predicted_zev'])/1e6,
            'Error (%)': ((rmse_results['actual_zev'] - rmse_results['predicted_zev'])/rmse_results['actual_zev'])*100
        })
        st.table(comparison_df.round(3))

    # Define scenarios
    scenarios = [
        ("Growth Focus", 0.7, 0.3),
        ("Incentive Focus", 0.3, 0.7),
        ("Balanced", 0.5, 0.5)
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
            
            # Plot historical total ZEVs
            historical_zev = results['historical_bev'] + results['historical_phev'] + results['historical_fcev']
            fig.add_trace(go.Scatter(
                x=results['historical_years'],
                y=historical_zev/1e6,
                name=f"{name} (Historical)",
                mode='lines',
                line=dict(width=1, dash='dot')
            ))
            
            # Plot future total ZEVs
            future_zev_mean = results['future_bev_mean'] + results['future_phev_mean'] + results['future_fcev_mean']
            future_zev_lower = results['future_bev_lower'] + results['future_phev_lower'] + results['future_fcev_lower']
            future_zev_upper = results['future_bev_upper'] + results['future_phev_upper'] + results['future_fcev_upper']
            
            fig.add_trace(go.Scatter(
                x=results['future_years'],
                y=future_zev_mean/1e6,
                name=f"{name} (Projected)",
                mode='lines',
                line=dict(width=2)
            ))
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=results['future_years'],
                y=future_zev_upper/1e6,
                name=f"{name} CI",
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=results['future_years'],
                y=future_zev_lower/1e6,
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
            title=f'ZEV Adoption Scenarios with Historical RMSE ({rmse_results["rmse_percentage"]*100:.1f}%)',
            xaxis_title='Year',
            yaxis_title='Vehicles (Millions)',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Error Impact Analysis")
        
        impact_data = []
        for name in scenario_results:
            result = scenario_results[name]
            final_zev_mean = (result['future_bev_mean'][-1] + 
                            result['future_phev_mean'][-1] + 
                            result['future_fcev_mean'][-1])/1e6
            final_zev_lower = (result['future_bev_lower'][-1] + 
                             result['future_phev_lower'][-1] + 
                             result['future_fcev_lower'][-1])/1e6
            final_zev_upper = (result['future_bev_upper'][-1] + 
                             result['future_phev_upper'][-1] + 
                             result['future_fcev_upper'][-1])/1e6
            uncertainty = final_zev_upper - final_zev_lower
            
            impact_data.append({
                'Scenario': name,
                'Mean 2027 ZEVs (M)': f"{final_zev_mean:.2f}",
                'Lower 90% CI (M)': f"{final_zev_lower:.2f}",
                'Upper 90% CI (M)': f"{final_zev_upper:.2f}",
                'Uncertainty Range (M)': f"{uncertainty:.2f}"
            })
        
        st.table(pd.DataFrame(impact_data))

    with tab4:
        st.header("Monte Carlo Distribution Analysis")
        
        selected_scenario = st.selectbox(
            "Select scenario for detailed analysis",
            list(scenario_results.keys())
        )
        
        selected_vehicle = st.selectbox(
            "Select vehicle type",
            ["Total ZEVs", "BEVs", "PHEVs", "FCEVs"]
        )
        
        result = scenario_results[selected_scenario]
        
        if selected_vehicle == "Total ZEVs":
            final_values = ((result['bev_simulations'][-1] + 
                           result['phev_simulations'][-1] + 
                           result['fcev_simulations'][-1])/1e6)
            mean_value = (result['future_bev_mean'][-1] + 
                         result['future_phev_mean'][-1] + 
                         result['future_fcev_mean'][-1])/1e6
            lower_bound = (result['future_bev_lower'][-1] + 
                          result['future_phev_lower'][-1] + 
                          result['future_fcev_lower'][-1])/1e6
            upper_bound = (result['future_bev_upper'][-1] + 
                          result['future_phev_upper'][-1] + 
                          result['future_fcev_upper'][-1])/1e6
        elif selected_vehicle == "BEVs":
            final_values = result['bev_simulations'][-1]/1e6
            mean_value = result['future_bev_mean'][-1]/1e6
            lower_bound = result['future_bev_lower'][-1]/1e6
            upper_bound = result['future_bev_upper'][-1]/1e6
        elif selected_vehicle == "PHEVs":
            final_values = result['phev_simulations'][-1]/1e6
            mean_value = result['future_phev_mean'][-1]/1e6
            lower_bound = result['future_phev_lower'][-1]/1e6
            upper_bound = result['future_phev_upper'][-1]/1e6
        else:  # FCEVs
            final_values = result['fcev_simulations'][-1]/1e6
            mean_value = result['future_fcev_mean'][-1]/1e6
            lower_bound = result['future_fcev_lower'][-1]/1e6
            upper_bound = result['future_fcev_upper'][-1]/1e6
        
        fig_hist = plt.figure(figsize=(10, 6))
        plt.hist(
            final_values,
            bins=50,
            density=True,
            alpha=0.7,
            color='skyblue',
            edgecolor='black'
        )
        plt.axvline(
            mean_value,
            color='red',
            linestyle='--',
            label='Mean'
        )
        plt.axvline(
            lower_bound,
            color='green',
            linestyle=':',
            label='90% CI'
        )
        plt.axvline(
            upper_bound,
            color='green',
            linestyle=':',
            label='90% CI'
        )
        plt.title(f"Distribution of 2027 {selected_vehicle} - {selected_scenario}")
        plt.xlabel("Number of Vehicles (Millions)")
        plt.ylabel("Density")
        plt.legend()
        st.pyplot(fig_hist)
        
        # Add numerical summary
        st.subheader("Distribution Statistics")
        st.write(f"Mean: {mean_value:.2f}M vehicles")
        st.write(f"90% Confidence Interval: [{lower_bound:.2f}M, {upper_bound:.2f}M]")
        st.write(f"Range of Uncertainty: {(upper_bound - lower_bound):.2f}M vehicles")

    # Sidebar information
    with st.sidebar:
        st.header("Analysis Parameters")
        st.write("Total Budget: $5 Billion")
        st.write("Period: 2024-2027")
        
        # Show initial conditions
        st.markdown("---")
        st.write("Initial Conditions (2024):")
        initial_zev = (V_BEV_data[-1] + V_PHEV_data[-1] + V_FCEV_data[-1])/1e6
        st.write(f"- Total ZEVs: {initial_zev:.2f}M")
        st.write(f"- BEVs: {V_BEV_data[-1]/1e6:.2f}M")
        st.write(f"- PHEVs: {V_PHEV_data[-1]/1e6:.2f}M")
        st.write(f"- FCEVs: {V_FCEV_data[-1]/1e6:.2f}M")
        
        st.markdown("---")
        
        st.write("Error Metrics:")
        st.write(f"- Historical RMSE: {rmse_results['rmse_percentage']*100:.2f}%")
        st.write(f"- Historical MAPE: {rmse_results['mape']:.2f}%")
        st.write("- Confidence Level: 90%")
        st.write("- Monte Carlo Runs: 1,000")
        
        st.markdown("---")
        st.write("Scenario Definitions:")
        for name, growth, incentive in scenarios:
            st.write(f"- {name}:")
            st.write(f"  * Growth: {growth*100}%")
            st.write(f"  * Incentives: {incentive*100}%")
        
        if st.button("Download Results"):
            # Create detailed results DataFrame
            results_df = pd.DataFrame()
            
            # Add years
            results_df['Year'] = list(scenario_results['Balanced']['historical_years']) + \
                                list(scenario_results['Balanced']['future_years'])
            
            # Add data for each scenario
            for name in scenario_results:
                result = scenario_results[name]
                
                # Historical data
                hist_zev = (result['historical_bev'] + 
                           result['historical_phev'] + 
                           result['historical_fcev'])/1e6
                
                # Future projections
                future_zev_mean = (result['future_bev_mean'] + 
                                 result['future_phev_mean'] + 
                                 result['future_fcev_mean'])/1e6
                future_zev_lower = (result['future_bev_lower'] + 
                                  result['future_phev_lower'] + 
                                  result['future_fcev_lower'])/1e6
                future_zev_upper = (result['future_bev_upper'] + 
                                  result['future_phev_upper'] + 
                                  result['future_fcev_upper'])/1e6
                
                # Combine historical and future data
                results_df[f"{name} (Historical)"] = list(hist_zev) + [None] * len(future_zev_mean)
                results_df[f"{name} (Projected)"] = [None] * len(hist_zev) + list(future_zev_mean)
                results_df[f"{name} (Lower CI)"] = [None] * len(hist_zev) + list(future_zev_lower)
                results_df[f"{name} (Upper CI)"] = [None] * len(hist_zev) + list(future_zev_upper)
            
            # Save to Excel
            results_df.to_excel('zev_scenarios_with_rmse.xlsx', index=False)
            st.success("Results downloaded successfully!")

except Exception as e:
    st.error(f"Error in analysis: {str(e)}")
    st.write("Please ensure all data files are available and properly formatted.")
