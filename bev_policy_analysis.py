import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import solve_ivp

# Set page configuration
st.set_page_config(layout="wide", page_title="BEV Policy Mix Analysis")

# Page Title
st.title("BEV Policy Mix Analysis for California (2024-2027)")
st.markdown("### Analysis of $5B Budget Allocation Across Different Policy Mixes")

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

    def create_policy_params(base_params):
        policy_params = base_params.copy()
        policy_params.update({
            'reg_strength': 0.15,
            'reg_compliance': 0.85,
            'pp_multiplier': 1.2,
            'private_investment': 0.3,
            'edu_effectiveness': 0.25,
            'awareness_decay': 0.1,
            'rd_improvement': 0.2,
            'rd_delay': 1.0,
        })
        return policy_params

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

    def run_policy_scenario(base_params, scenario_config, total_budget=5e9, years=4):
        scenario_params = create_policy_params(base_params)
        annual_budget = total_budget / years
        
        # Calculate budget allocations
        reg_budget = annual_budget * scenario_config['regulatory']
        pp_budget = annual_budget * scenario_config['partnership']
        edu_budget = annual_budget * scenario_config['education']
        rd_budget = annual_budget * scenario_config['rd']
        
        # Apply policy effects
        reg_scaling = 1 + (reg_budget/1e9) * scenario_params['reg_strength'] * \
                     scenario_params['reg_compliance']
        pp_scaling = 1 + (pp_budget/1e9) * scenario_params['pp_multiplier'] * \
                    (1 + scenario_params['private_investment'])
        edu_scaling = 1 + (edu_budget/1e9) * scenario_params['edu_effectiveness'] * \
                     np.exp(-scenario_params['awareness_decay'])
        rd_scaling = 1 + (rd_budget/1e9) * scenario_params['rd_improvement'] * \
                    (1 - np.exp(-years/scenario_params['rd_delay']))
        
        # Modify parameters
        scenario_params['k_C'] *= reg_scaling
        scenario_params['k_V'] *= reg_scaling
        scenario_params['r2'] *= (pp_scaling * rd_scaling)
        scenario_params['beta1'] *= (pp_scaling * edu_scaling)
        
        # Initial conditions for 2024 (1.7M BEVs)
        X0_2024 = [
            V_ICE_norm[0],
            1.7e6/V_BEV_mean,
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

    # Define policy scenarios
    policy_scenarios = {
        "Balanced Mix": {
            'regulatory': 0.25,
            'partnership': 0.25,
            'education': 0.25,
            'rd': 0.25
        },
        "Regulatory Focus": {
            'regulatory': 0.4,
            'partnership': 0.2,
            'education': 0.2,
            'rd': 0.2
        },
        "Partnership Focus": {
            'regulatory': 0.2,
            'partnership': 0.4,
            'education': 0.2,
            'rd': 0.2
        },
        "Education Focus": {
            'regulatory': 0.2,
            'partnership': 0.2,
            'education': 0.4,
            'rd': 0.2
        },
        "R&D Focus": {
            'regulatory': 0.2,
            'partnership': 0.2,
            'education': 0.2,
            'rd': 0.4
        }
    }

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview & Results",
        "Detailed Analysis",
        "Policy Components",
        "Recommendations"
    ])

    with tab1:
        st.header("Overview of Policy Mix Scenarios")
        st.markdown("""
        This analysis explores different approaches to allocating a $5 billion budget across
        four major policy components:
        - Regulatory measures
        - Public-private partnerships
        - Education and awareness programs
        - Research and development initiatives
        """)

        # Run scenarios and create visualization
        fig = go.Figure()
        scenario_results = {}

        for name, config in policy_scenarios.items():
            years, bev = run_policy_scenario(params, config)
            scenario_results[name] = (years, bev)
            fig.add_trace(go.Scatter(
                x=years,
                y=bev/1e6,
                name=name,
                mode='lines+markers'
            ))

        fig.update_layout(
            title='BEV Adoption Trajectories by Policy Mix',
            xaxis_title='Year',
            yaxis_title='BEV Vehicles (Millions)',
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Detailed Analysis")
        
        # Create metrics
        scenario_metrics = []
        for name, (years, bev) in scenario_results.items():
            final_adoption = bev[-1]/1e6
            improvement = (bev[-1] - 1.7e6) / 1.7e6 * 100
            annual_growth = (np.power(bev[-1]/1.7e6, 1/4) - 1) * 100
            
            scenario_metrics.append({
                'Scenario': name,
                'Final Adoption (M)': f"{final_adoption:.2f}",
                'Improvement (%)': f"{improvement:.1f}",
                'Annual Growth (%)': f"{annual_growth:.1f}"
            })
        
        st.table(pd.DataFrame(scenario_metrics))

        selected_scenario = st.selectbox(
            "Select a scenario for detailed analysis:",
            list(policy_scenarios.keys())
        )

        st.subheader(f"Budget Allocation - {selected_scenario}")
        
        # Show budget allocation
        fig_pie = px.pie(
            values=list(policy_scenarios[selected_scenario].values()),
            names=list(policy_scenarios[selected_scenario].keys()),
            title=f"Budget Distribution - {selected_scenario}"
        )
        st.plotly_chart(fig_pie)

    with tab3:
        st.header("Policy Components Analysis")
        
        st.markdown("""
        ### Component Effects
        
        1. **Regulatory Measures**
        - Compliance requirements
        - Standards enforcement
        - Policy strength effects
        
        2. **Public-Private Partnerships**
        - Private sector leverage
        - Investment multiplier effects
        - Collaborative initiatives
        
        3. **Education & Awareness**
        - Public awareness campaigns
        - Consumer education
        - Information dissemination
        
        4. **Research & Development**
        - Technology improvements
        - Innovation effects
        - Time-delayed benefits
        """)

    with tab4:
        st.header("Strategic Recommendations")
        
        # Find best performing scenario
        best_scenario = max(
            scenario_results.items(),
            key=lambda x: x[1][1][-1]
        )
        
        st.markdown(f"""
        ### Key Findings
        
        1. **Most Effective Strategy**: {best_scenario[0]}
            - Achieves highest BEV adoption by 2027
            - Shows most consistent growth trajectory
        
        2. **Implementation Recommendations**:
            - Phase 1 (2024-2025): Focus on infrastructure and regulatory framework
            - Phase 2 (2026-2027): Scale up education and R&D initiatives
        
        3. **Critical Success Factors**:
            - Policy consistency
            - Stakeholder engagement
            - Regular monitoring and adjustment
            - Cross-sector collaboration
        """)

    # Sidebar
    with st.sidebar:
        st.header("Analysis Parameters")
        st.write("Total Budget: $5 Billion")
        st.write("Timeframe: 2024-2027")
        st.write("Initial BEV Fleet: 1.7M")
        
        st.markdown("---")
        
        st.header("Model Parameters")
        if st.checkbox("Show Policy Parameters"):
            st.write(create_policy_params(params))

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.write("Please ensure ZEVdata.xlsx is in the correct location.")