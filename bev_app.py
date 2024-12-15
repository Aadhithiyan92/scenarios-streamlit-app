import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# Set page config
st.set_page_config(layout="wide", page_title="BEV Adoption Scenarios Analysis")

# Page title
st.title("BEV Adoption Scenarios Analysis (2024-2027)")
st.write("Analysis of $5B Budget Allocation for California's BEV Adoption")

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
        
        total_vehicles = V + B
        ev_fraction = B / total_vehicles if total_vehicles > 0 else 0

        # System equations
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

    def run_scenario(base_params, growth_ratio, incentive_ratio, total_budget=5e9, years=4):
        """Run a scenario with given growth and incentive ratios"""
        scenario_params = base_params.copy()
        
        # Calculate annual budget allocation
        annual_budget = total_budget / years
        growth_budget = annual_budget * growth_ratio
        incentive_budget = annual_budget * incentive_ratio
        
        # Modify parameters based on budget allocation
        growth_scaling = 1 + (growth_budget/1e9) * 0.15
        incentive_scaling = 1 + (incentive_budget/1e9) * 0.20
        
        scenario_params['r2'] *= growth_scaling
        scenario_params['k_C'] *= incentive_scaling
        scenario_params['k_V'] *= incentive_scaling
        scenario_params['beta1'] *= incentive_scaling
        
        # Initial conditions for 2024 (1.7M BEVs)
        X0_2024 = [
            V_ICE_norm[0],
            1e6/V_BEV_mean,
            M_norm[0],
            C_norm[0],
            S_norm[0]
        ]
        
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

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Scenario Comparison",
        "Detailed Analysis",
        "Recommendations"
    ])

    # Define scenarios
    scenarios = [
        ("Early Growth Push", 0.7, 0.3),
        ("Balanced Progression", 0.5, 0.5),
        ("Market Acceleration", 0.4, 0.6),
        ("Infrastructure First", 0.8, 0.2),
        ("Consumer Incentive Priority", 0.3, 0.7)
    ]

    with tab1:
        st.header("Scenario Comparison")
        
        # Run scenarios and create visualization
        fig = go.Figure()
        scenario_results = {}

        for name, growth_ratio, incentive_ratio in scenarios:
            years, bev = run_scenario(params, growth_ratio, incentive_ratio)
            scenario_results[name] = (years, bev)
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

    with tab2:
        st.header("Detailed Analysis")
        
        selected_scenario = st.selectbox(
            "Select a scenario for detailed analysis:",
            [name for name, _, _ in scenarios]
        )
        
        # Display scenario details
        scenario_info = next((s for s in scenarios if s[0] == selected_scenario))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Budget Allocation")
            st.write(f"Growth Ratio: {scenario_info[1]:.1%}")
            st.write(f"Incentive Ratio: {scenario_info[2]:.1%}")
            
            # Calculate annual budget
            annual_budget = 5e9 / 4
            st.write(f"Annual Growth Budget: ${scenario_info[1] * annual_budget/1e9:.1f}B")
            st.write(f"Annual Incentive Budget: ${scenario_info[2] * annual_budget/1e9:.1f}B")
        
        with col2:
            st.subheader("Projected Outcomes")
            years, bev = scenario_results[selected_scenario]
            final_adoption = bev[-1]
            improvement = (final_adoption - 1e6) / 1e6 * 100
            
            st.metric("Final Adoption (2027)", f"{final_adoption/1e6:.2f}M vehicles")
            st.metric("Improvement from 2024", f"{improvement:.1f}%")
            st.metric("Annual Growth Rate", 
                     f"{(np.power(final_adoption/1e6, 1/4) - 1)*100:.1f}%")

    with tab3:
        st.header("Strategic Recommendations")
        
        # Find best performing scenario
        best_scenario = max(scenario_results.items(), key=lambda x: x[1][1][-1])
        
        st.markdown(f"""
        ### Key Findings
        
        1. **Most Effective Strategy**: {best_scenario[0]}
           - Achieves highest BEV adoption by 2027
           - Shows most consistent growth trajectory
        
        2. **Implementation Recommendations**:
           - Focus on infrastructure development early
           - Scale incentives based on market readiness
           - Monitor and adjust based on adoption rates
        
        3. **Critical Success Factors**:
           - Consistent policy implementation
           - Regular progress monitoring
           - Stakeholder engagement
        """)

    # Sidebar
    with st.sidebar:
        st.header("Analysis Parameters")
        st.write("Total Budget: $5 Billion")
        st.write("Timeframe: 2024-2027")
        st.write("Initial BEV Fleet: 1.7M")
        
        if st.button("Download Results"):
            results_df = pd.DataFrame({
                'Year': years,
                **{name: bev/1e6 for name, (years, bev) in scenario_results.items()}
            })
            results_df.to_excel('bev_scenario_results_2027.xlsx', index=False)
            st.success("Results downloaded as 'bev_scenario_results_2027.xlsx'")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.write("Please ensure ZEVdata.xlsx is in the correct location.")