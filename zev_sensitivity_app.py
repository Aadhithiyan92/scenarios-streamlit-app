import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (adjust the path as needed)
data = pd.read_excel('C:/Users/as3889/Desktop/ZEVdata.xlsx')
years = data['Year'].values
V_ICE_data = data['V'].values
V_BEV_data = data['BEV'].values
V_PHEV_data = data['PHEV'].values
V_FCEV_data = data['FCEV'].values
M_data = data['M'].values
C_data = data['C'].values
S_data = data['S'].values

# Normalize the data using mean values
V_ICE_mean, V_BEV_mean, V_PHEV_mean, V_FCEV_mean = np.mean(V_ICE_data), np.mean(V_BEV_data), np.mean(V_PHEV_data), np.mean(V_FCEV_data)
M_mean, C_mean, S_mean = np.mean(M_data), np.mean(C_data), np.mean(S_data)

V_ICE_norm = V_ICE_data / V_ICE_mean
V_BEV_norm = V_BEV_data / V_BEV_mean
V_PHEV_norm = V_PHEV_data / V_PHEV_mean
V_FCEV_norm = V_FCEV_data / V_FCEV_mean
M_norm = M_data / M_mean
C_norm = C_data / C_mean
S_norm = S_data / S_mean

# Base year for time calculation
base_year = 2010

# Initial conditions
X0 = [V_ICE_norm[0], V_BEV_norm[0], V_PHEV_norm[0], V_FCEV_norm[0], M_norm[0], C_norm[0], S_norm[0]]

# Incentive calculations
total_incentives = 1511901636 + 25180906 + 121114580 + 2685500
total_vehicles = (426921 + 152330 + 14305) + (3749 + 1121 + 68) + (3597 + 9648 + 144) + (221 + 279 + 7)
mean_incentive = total_incentives / total_vehicles

cvrp_incentive_norm = (1511901636 / (426921 + 152330 + 14305)) / mean_incentive
cvap_incentive_norm = (25180906 / (3749 + 1121 + 68)) / mean_incentive
cc4a_incentive_norm = (121114580 / (3597 + 9648 + 144)) / mean_incentive
dcap_incentive_norm = (2685500 / (221 + 279 + 7)) / mean_incentive

# Calculate average of known incentives
known_incentives_norm = [cvrp_incentive_norm, cvap_incentive_norm, cc4a_incentive_norm, dcap_incentive_norm]
avg_known_incentive_norm = sum(known_incentives_norm) / len(known_incentives_norm)

# Estimate other incentives (e.g., federal tax credits)
federal_incentive_total = 2500 * total_vehicles  # Assuming $7000 tax credit for all EVs
other_incentive_norm = (federal_incentive_total / total_vehicles) / mean_incentive

# Model parameters
params = {
    'r1': 0.04600309537728457,
    'K1': 19.988853430783674,
    'alpha1': 2.1990994330017567e-20,
    'alpha2': 7.217511478131137,
    'alpha3': 8.363375552715492,
    'r2': 0.4272847858514477,
    'beta1': 0.0009,
    'gamma1': 0.006195813413796029,
    'r3': 0.11,
    'beta2': 0.0003,
    'gamma2': 0.07,
    'r4': 0.14351479815351475,
    'beta3': 0.0002,
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
    'k_O': 0.4143805321682873,
    'lambda_C': 0.1699545021325927,
    'lambda_V': 0.16153300196227824,
    'lambda_A': 0.16300789498583168,
    'lambda_D': 0.16186856013035358,
    'lambda_O': 0.165008529345057,
    'kappa': 0.3673510495799293,
    'lambda_S': 0.004719196117653555,
    'omega': 0.0773179311036966,
    'tau': 0.04,
    'cvrp_end_year': 2023  # Default end year for CVRP
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

    # Unpack parameters
    r1, K1, alpha1, alpha2, alpha3, r2, beta1, gamma1, r3, beta2, gamma2, r4, beta3, gamma3, \
    phi1, phi2, phi3, phi4, eta, psi1, psi2, psi3, psi4, delta, epsilon, zeta, \
    k_C, k_V, k_A, k_D, k_O, lambda_C, lambda_V, lambda_A, lambda_D, lambda_O, \
    kappa, lambda_S, omega, tau, cvrp_end_year = params.values()

    # Calculate incentive effects
    cvrp_effect = incentive_time_effect(t + base_year, 2010, cvrp_end_year, k_C * cvrp_incentive_norm, lambda_C)
    cvap_effect = incentive_time_effect(t + base_year, 2018, 2030, k_V * cvap_incentive_norm, lambda_V)
    cc4a_effect = incentive_time_effect(t + base_year, 2015, 2030, k_A * cc4a_incentive_norm, lambda_A)
    dcap_effect = incentive_time_effect(t + base_year, 2016, 2030, k_D * dcap_incentive_norm, lambda_D)
    other_effect = k_O * other_incentive_norm * np.exp(-lambda_O * t)

    total_incentive = cvrp_effect + cvap_effect + cc4a_effect + dcap_effect + other_effect
    total_vehicles = V + B + P + F
    ev_fraction = (B + P + F) / total_vehicles if total_vehicles > 0 else 0

    # System equations
    dV_dt = r1 * V * (1 - total_vehicles/K1) * (1 - omega * ev_fraction) - tau * V * ev_fraction - epsilon * V
    dB_dt = r2 * B + beta1 * total_incentive + alpha1 * tau * V * ev_fraction - gamma1 * B
    dP_dt = r3 * P + beta2 * total_incentive + alpha2 * tau * V * ev_fraction - gamma2 * P
    dF_dt = r4 * F + beta3 * total_incentive + alpha3 * tau * V * ev_fraction - gamma3 * F
    dM_dt = phi1 * V + phi2 * B + phi3 * P + phi4 * F - eta * M
    dC_dt = (psi1 * V + psi2 * B + psi3 * P + psi4 * F) * M / total_vehicles - delta * C + zeta * (V / total_vehicles) ** 2
    dS_dt = kappa * (B + P + F) / total_vehicles - lambda_S * S

    return [dV_dt, dB_dt, dP_dt, dF_dt, dM_dt, dC_dt, dS_dt]

def project_ev_adoption(projection_year, params):
    """Project EV adoption up to the specified year"""
    # Initial conditions
    X0 = [V_ICE_norm[0], V_BEV_norm[0], V_PHEV_norm[0], V_FCEV_norm[0], M_norm[0], C_norm[0], S_norm[0]]

    # Time array for projection
    t_span = (0, projection_year - base_year)
    t_eval = np.arange(0, projection_year - base_year + 1, 1)

    # Solve the system
    solution = solve_ivp(system, t_span, X0, args=(params,), t_eval=t_eval, method='RK45')

    # Denormalize the results
    V_ICE_proj = solution.y[0] * V_ICE_mean
    V_BEV_proj = solution.y[1] * V_BEV_mean
    V_PHEV_proj = solution.y[2] * V_PHEV_mean
    V_FCEV_proj = solution.y[3] * V_FCEV_mean
    C_proj = solution.y[5] * C_mean
    

    # Generate years for x-axis
    projection_years_array = np.arange(base_year, projection_year + 1)
    

    return projection_years_array, V_ICE_proj, V_BEV_proj, V_PHEV_proj, V_FCEV_proj, C_proj
def create_incentive_sensitivity_params():
    """Create parameter sets for incentive sensitivity analysis"""
    
    # Base parameters
    base_params = params.copy()
    base_k_C = 9.466080903837223  # CVRP
    base_k_V = 0.1743857370252369 # CVAP
    base_k_A = 0.9252378762028231 # CC4A
    base_k_D = 0.282707168138512  # DCAP
    
    # Create variations
    variations = {
        'Low Incentives (-20%)': 0.80,
        'Moderate Incentives (-10%)': 0.90,
        'Base Incentives': 1.0,
        'High Incentives (+10%)': 1.10,
        'Very High Incentives (+20%)': 1.20
    }
    
    sensitivity_params = {}
    
    for variation_name, factor in variations.items():
        params_variation = base_params.copy()
        params_variation.update({
            'k_C': base_k_C * factor,
            'k_V': base_k_V * factor,
            'k_A': base_k_A * factor,
            'k_D': base_k_D * factor
        })
        sensitivity_params[variation_name] = params_variation
    
    return sensitivity_params

def create_growth_sensitivity_params():
    """Create parameter sets for growth rate sensitivity analysis"""
    
    # Base parameters
    base_params = params.copy()
    base_r2 = 0.4272847858514477  # BEV
    base_r3 = 0.11                # PHEV
    base_r4 = 0.14351479815351475 # FCEV
    
    variations = {
        'Low Growth (-10%)': 0.90,
        'Moderate Growth (-5%)': 0.95,
        'Base Growth': 1.0,
        'High Growth (+5%)': 1.05,
        'Very High Growth (+10%)': 1.10
    }
    
    sensitivity_params = {}
    
    for variation_name, factor in variations.items():
        params_variation = base_params.copy()
        params_variation.update({
            'r2': base_r2 * factor,
            'r3': base_r3 * factor,
            'r4': base_r4 * factor
        })
        sensitivity_params[variation_name] = params_variation
    
    return sensitivity_params

def analyze_sensitivity_results(scenarios, analysis_type):
    """Analyze and compare sensitivity analysis results"""
    results_list = []  # Create a list to store dictionaries of results
    
    for scenario_name, data in scenarios.items():
        for year_idx, year in enumerate(data['years']):
            total = (data['ICE'][year_idx] + data['BEV'][year_idx] + 
                    data['PHEV'][year_idx] + data['FCEV'][year_idx])
            
            total_zev = (data['BEV'][year_idx] + data['PHEV'][year_idx] + 
                        data['FCEV'][year_idx])
            
            # Create a dictionary for each row
            row_dict = {
                'Scenario': scenario_name,
                'Year': year,
                'BEV_Count': data['BEV'][year_idx],
                'BEV_Percent': data['BEV'][year_idx]/total*100,
                'PHEV_Count': data['PHEV'][year_idx],
                'PHEV_Percent': data['PHEV'][year_idx]/total*100,
                'FCEV_Count': data['FCEV'][year_idx],
                'FCEV_Percent': data['FCEV'][year_idx]/total*100,
                'Total_ZEV': total_zev,
                'ZEV_Percent': total_zev/total*100,
                'ICE_Count': data['ICE'][year_idx],
                'ICE_Percent': data['ICE'][year_idx]/total*100
            }
            results_list.append(row_dict)
    
    # Create DataFrame from list of dictionaries
    results_df = pd.DataFrame(results_list)
    return results_df

def plot_sensitivity_comparison(scenarios, analysis_type):
    """Create plots for sensitivity analysis comparison"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = sns.color_palette("husl", len(scenarios))
    
    for (scenario_name, data), color in zip(scenarios.items(), colors):
        # BEV adoptions
        ax1.plot(data['years'], data['BEV'], 
                label=scenario_name, color=color, marker='o', markersize=4)
        
        # Total ZEV percentage
        total_zev = [data['BEV'][i] + data['PHEV'][i] + data['FCEV'][i] 
                     for i in range(len(data['years']))]
        total_fleet = [total_zev[i] + data['ICE'][i] 
                      for i in range(len(data['years']))]
        zev_percentage = [total_zev[i]/total_fleet[i]*100 
                         for i in range(len(data['years']))]
        
        ax2.plot(data['years'], zev_percentage, 
                label=scenario_name, color=color, marker='o', markersize=4)
    
    ax1.set_title(f'BEV Adoption {analysis_type} Sensitivity')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of BEVs')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax2.set_title(f'Total ZEV Percentage {analysis_type} Sensitivity')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('ZEV Percentage of Total Fleet')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def main():
    st.title('ZEV Adoption Sensitivity Analysis')
    
    # Sidebar for user inputs
    st.sidebar.header('Analysis Parameters')
    
    analysis_type = st.sidebar.selectbox(
        'Select Analysis Type',
        ['Growth Rate Sensitivity', 'Incentive Sensitivity']
    )
    
    projection_year = st.sidebar.slider(
        'Projection Year',
        min_value=2023,
        max_value=2030,
        value=2027
    )
    
    cvrp_continues = st.sidebar.checkbox(
        'Continue CVRP after 2023',
        value=True
    )
    
    # Main content
    st.header(f'{analysis_type} Analysis')
    
    try:
        # Create appropriate parameter sets based on analysis type
        if analysis_type == 'Growth Rate Sensitivity':
            sensitivity_params = create_growth_sensitivity_params()
        else:
            sensitivity_params = create_incentive_sensitivity_params()
        
        scenarios = {}
        
        # Generate projections for each scenario
        for scenario_name, scenario_params in sensitivity_params.items():
            if cvrp_continues:
                scenario_params['cvrp_end_year'] = projection_year
            else:
                scenario_params['cvrp_end_year'] = 2023
            
            years, ICE, BEV, PHEV, FCEV, CO2 = project_ev_adoption(
                projection_year, 
                scenario_params
            )
            
            scenarios[scenario_name] = {
                'years': years,
                'ICE': ICE,
                'BEV': BEV,
                'PHEV': PHEV,
                'FCEV': FCEV
            }
        
        # Display results
        results_df = analyze_sensitivity_results(scenarios, analysis_type)
        
        # Show summary statistics
        st.subheader('Summary Statistics')
        summary_year = st.selectbox('Select Year for Summary', 
                                  sorted(results_df['Year'].unique()))
        
        year_summary = results_df[results_df['Year'] == summary_year]
        st.dataframe(year_summary[['Scenario', 'BEV_Count', 'PHEV_Count', 
                                 'FCEV_Count', 'Total_ZEV', 'ZEV_Percent']])
        
        # Display visualization
        st.subheader('Visualization')
        fig = plot_sensitivity_comparison(scenarios, analysis_type)
        st.pyplot(fig)
        
        # Add download button for results
        st.subheader('Download Results')
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f'zev_{analysis_type.lower().replace(" ", "_")}_{projection_year}.csv',
            mime='text/csv'
        )
        
        # Display additional analysis metrics
        st.subheader('Key Metrics')
        final_year = results_df[results_df['Year'] == projection_year]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Average ZEV Percentage",
                f"{final_year['ZEV_Percent'].mean():.1f}%",
                f"{final_year['ZEV_Percent'].std():.1f}% std dev"
            )
            
        with col2:
            st.metric(
                "Total ZEV Range",
                f"{final_year['Total_ZEV'].min():,.0f} - {final_year['Total_ZEV'].max():,.0f}",
                f"Spread: {final_year['Total_ZEV'].max() - final_year['Total_ZEV'].min():,.0f}"
            )
        
        # Add comparison between scenarios
        st.subheader('Scenario Comparison')
        comparison_metrics = final_year.pivot(columns='Scenario', 
                                           values=['BEV_Count', 'Total_ZEV', 'ZEV_Percent'])
        st.dataframe(comparison_metrics)
        
        # Add interactive chart selection
        chart_type = st.selectbox(
            'Select Chart Type',
            ['BEV Adoption', 'Total ZEV Percentage', 'Vehicle Type Distribution']
        )
        
        if chart_type == 'Vehicle Type Distribution':
            # Create stacked bar chart for vehicle distribution
            fig_dist = plt.figure(figsize=(12, 6))
            scenario_names = results_df['Scenario'].unique()
            x = np.arange(len(scenario_names))
            width = 0.35
            
            latest_data = results_df[results_df['Year'] == projection_year]
            
            plt.bar(x, latest_data['BEV_Percent'], width, label='BEV')
            plt.bar(x, latest_data['PHEV_Percent'], width, 
                   bottom=latest_data['BEV_Percent'], label='PHEV')
            plt.bar(x, latest_data['FCEV_Percent'], width,
                   bottom=latest_data['BEV_Percent'] + latest_data['PHEV_Percent'],
                   label='FCEV')
            
            plt.xlabel('Scenarios')
            plt.ylabel('Percentage')
            plt.title(f'Vehicle Distribution by Scenario ({projection_year})')
            plt.xticks(x, scenario_names, rotation=45)
            plt.legend()
            
            st.pyplot(fig_dist)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your input parameters and try again.")

if __name__ == "__main__":
    main()