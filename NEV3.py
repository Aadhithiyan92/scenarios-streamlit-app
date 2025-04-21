import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
import json
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Nationwide ZEV Adoption Modeling",
    page_icon="🚗",
    layout="wide"
)

# App title and description
st.title("Nationwide ZEV Adoption Modeling Dashboard")
st.markdown("""
This dashboard implements a modular state-space modeling approach for analyzing Zero Emission Vehicle (ZEV) 
adoption across multiple states. Each state is modeled independently with state-specific characteristics including 
incentive structures, infrastructure investments, demographic trends, and policy targets.
""")

# Initialize session state for storing data and model parameters
if 'states_data' not in st.session_state:
    # Sample data for initial states (would be replaced with actual data)
    st.session_state.states_data = {
        'California': {
            'ICEV_initial': 30000000,
            'BEV_initial': 1000000,
            'PHEV_initial': 500000,
            'FCEV_initial': 10000,
            'charging_stations': 80000,
            'incentives': 7500,
            'infrastructure_budget': 2.5,
            'policy_stringency': 0.8
        },
        'New York': {
            'ICEV_initial': 18000000,
            'BEV_initial': 250000,
            'PHEV_initial': 150000,
            'FCEV_initial': 2000,
            'charging_stations': 35000,
            'incentives': 2000,
            'infrastructure_budget': 1.8,
            'policy_stringency': 0.6
        },
        'Texas': {
            'ICEV_initial': 25000000,
            'BEV_initial': 120000,
            'PHEV_initial': 80000,
            'FCEV_initial': 500,
            'charging_stations': 20000,
            'incentives': 1000,
            'infrastructure_budget': 1.0,
            'policy_stringency': 0.3
        },
        'Florida': {
            'ICEV_initial': 20000000,
            'BEV_initial': 100000,
            'PHEV_initial': 70000,
            'FCEV_initial': 300,
            'charging_stations': 15000,
            'incentives': 1500,
            'infrastructure_budget': 0.8,
            'policy_stringency': 0.4
        },
        'Washington': {
            'ICEV_initial': 6500000,
            'BEV_initial': 150000,
            'PHEV_initial': 50000,
            'FCEV_initial': 1000,
            'charging_stations': 25000,
            'incentives': 5000,
            'infrastructure_budget': 1.5,
            'policy_stringency': 0.7
        },
        'Massachusetts': {
            'ICEV_initial': 5500000,
            'BEV_initial': 90000,
            'PHEV_initial': 45000,
            'FCEV_initial': 500,
            'charging_stations': 18000,
            'incentives': 4000,
            'infrastructure_budget': 1.2,
            'policy_stringency': 0.65
        },
        'Colorado': {
            'ICEV_initial': 5000000,
            'BEV_initial': 70000,
            'PHEV_initial': 30000,
            'FCEV_initial': 200,
            'charging_stations': 10000,
            'incentives': 3500,
            'infrastructure_budget': 1.0,
            'policy_stringency': 0.5
        },
        'Oregon': {
            'ICEV_initial': 3500000,
            'BEV_initial': 60000,
            'PHEV_initial': 25000,
            'FCEV_initial': 300,
            'charging_stations': 12000,
            'incentives': 3000,
            'infrastructure_budget': 0.9,
            'policy_stringency': 0.6
        },
        'New Jersey': {
            'ICEV_initial': 7000000,
            'BEV_initial': 80000,
            'PHEV_initial': 40000,
            'FCEV_initial': 100,
            'charging_stations': 15000,
            'incentives': 2500,
            'infrastructure_budget': 1.1,
            'policy_stringency': 0.55
        },
        'Maryland': {
            'ICEV_initial': 4500000,
            'BEV_initial': 50000,
            'PHEV_initial': 30000,
            'FCEV_initial': 50,
            'charging_stations': 9000,
            'incentives': 2000,
            'infrastructure_budget': 0.8,
            'policy_stringency': 0.5
        },
        'Illinois': {
            'ICEV_initial': 10000000,
            'BEV_initial': 75000,
            'PHEV_initial': 35000,
            'FCEV_initial': 100,
            'charging_stations': 12000,
            'incentives': 2000,
            'infrastructure_budget': 0.9,
            'policy_stringency': 0.45
        },
        'Pennsylvania': {
            'ICEV_initial': 9500000,
            'BEV_initial': 55000,
            'PHEV_initial': 30000,
            'FCEV_initial': 50,
            'charging_stations': 10000,
            'incentives': 1800,
            'infrastructure_budget': 0.7,
            'policy_stringency': 0.4
        },
        'Ohio': {
            'ICEV_initial': 9000000,
            'BEV_initial': 40000,
            'PHEV_initial': 25000,
            'FCEV_initial': 20,
            'charging_stations': 8000,
            'incentives': 1500,
            'infrastructure_budget': 0.6,
            'policy_stringency': 0.35
        },
        'Michigan': {
            'ICEV_initial': 8000000,
            'BEV_initial': 45000,
            'PHEV_initial': 30000,
            'FCEV_initial': 40,
            'charging_stations': 9000,
            'incentives': 2200,
            'infrastructure_budget': 0.8,
            'policy_stringency': 0.5
        },
        'Georgia': {
            'ICEV_initial': 8500000,
            'BEV_initial': 60000,
            'PHEV_initial': 35000,
            'FCEV_initial': 30,
            'charging_stations': 11000,
            'incentives': 1800,
            'infrastructure_budget': 0.7,
            'policy_stringency': 0.45
        }
    }

# Initialize nodes data for network visualization if not already present
if 'nodes_data' not in st.session_state:
    # Generate data based on state_data
    st.session_state.nodes_data = {}
    for state, data in st.session_state.states_data.items():
        zev_population = data['BEV_initial'] + data['PHEV_initial'] + data['FCEV_initial']
        st.session_state.nodes_data[state] = {
            'zev_population': zev_population,
            'charging_stations': data['charging_stations'],
            'avg_utilization': np.random.uniform(0.3, 0.8),  # Random for demonstration
            'fast_chargers_pct': np.random.uniform(0.1, 0.4),  # Random for demonstration
            'corridor_connections': np.random.randint(1, 5)  # Random for demonstration
        }

# State centroids for node map (approximate lat/long coordinates for state centers)
state_coordinates = {
    'California': [36.7783, -119.4179],
    'New York': [42.9538, -75.5268],
    'Texas': [31.9686, -99.9018],
    'Florida': [27.6648, -81.5158],
    'Washington': [47.7511, -120.7401],
    'Massachusetts': [42.4072, -71.3824],
    'Colorado': [39.5501, -105.7821],
    'Oregon': [43.8041, -120.5542],
    'New Jersey': [40.0583, -74.4057],
    'Maryland': [39.0458, -76.6413],
    'Illinois': [40.6331, -89.3985],
    'Pennsylvania': [41.2033, -77.1945],
    'Ohio': [40.4173, -82.9071],
    'Michigan': [44.3148, -85.6024],
    'Georgia': [33.0406, -83.6431]
}

# Function to model state-specific vehicle adoption dynamics
def state_dynamics(x, t, incentives, infrastructure, policy):
    """
    State-space model for vehicle population dynamics
    x[0]: ICEV (Internal Combustion Engine Vehicle) population
    x[1]: BEV (Battery Electric Vehicle) population
    x[2]: PHEV (Plug-in Hybrid Electric Vehicle) population
    x[3]: FCEV (Fuel Cell Electric Vehicle) population
    """
    # Parameters (would be calibrated with actual data)
    natural_replacement_rate = 0.1  # Base rate of vehicle replacement
    incentive_effectiveness = 0.02 * incentives / 5000  # Effect of incentives
    infrastructure_effect = 0.015 * infrastructure / 1.5  # Effect of infrastructure
    policy_effect = 0.025 * policy  # Effect of policy stringency
    
    # Combined effect of external forces
    external_factor = incentive_effectiveness + infrastructure_effect + policy_effect
    
    # Transition rates
    icev_to_bev_rate = 0.05 * (1 + external_factor)
    icev_to_phev_rate = 0.03 * (1 + external_factor)
    icev_to_fcev_rate = 0.005 * (1 + external_factor)
    
    # Differential equations
    icev_change = -natural_replacement_rate * x[0] * (icev_to_bev_rate + icev_to_phev_rate + icev_to_fcev_rate)
    bev_change = natural_replacement_rate * x[0] * icev_to_bev_rate + 0.01 * x[2]  # Some PHEVs convert to BEVs
    phev_change = natural_replacement_rate * x[0] * icev_to_phev_rate - 0.01 * x[2]
    fcev_change = natural_replacement_rate * x[0] * icev_to_fcev_rate
    
    return [icev_change, bev_change, phev_change, fcev_change]

# Function to calculate CO2 emissions
def calculate_emissions(vehicle_populations):
    # Emissions factors (tons CO2 per vehicle per year)
    icev_emissions_factor = 4.6
    phev_emissions_factor = 2.3
    bev_emissions_factor = 0
    fcev_emissions_factor = 0
    
    icev_total = vehicle_populations[:, 0] * icev_emissions_factor
    phev_total = vehicle_populations[:, 2] * phev_emissions_factor
    
    return icev_total + phev_total

# Function to create node-based geographical map
def create_zev_node_map():
    """
    Creates a node-based geographical map for visualizing EV data across states
    with interactive features showing connections and infrastructure.
    """
    st.header("ZEV Infrastructure Network Map")
    st.markdown("""
    This map visualizes the ZEV adoption and charging infrastructure as an interconnected network
    of nodes. The size of each node represents the ZEV population, while connections between states
    indicate infrastructure corridors and regional adoption patterns.
    """)
    
    # User controls
    col1, col2 = st.columns(2)
    
    with col1:
        node_size_metric = st.selectbox(
            "Node Size Represents",
            ["ZEV Population", "Charging Stations", "Fast Chargers"],
            index=0
        )
    
    with col2:
        connection_threshold = st.slider(
            "Connection Strength Threshold",
            0.0, 1.0, 0.3,
            help="Threshold for showing connections between states"
        )
    
    # Display options
    show_labels = st.checkbox("Show State Labels", value=True)
    show_metrics = st.checkbox("Show Metrics on Hover", value=True)
    
    # Create dataframe for map visualization
    nodes_df = pd.DataFrame([
        {
            'state': state,
            'lat': coords[0],
            'lon': coords[1],
            'zev_population': st.session_state.nodes_data[state]['zev_population'],
            'charging_stations': st.session_state.nodes_data[state]['charging_stations'],
            'fast_chargers': int(st.session_state.nodes_data[state]['charging_stations'] * 
                               st.session_state.nodes_data[state]['fast_chargers_pct']),
            'utilization': st.session_state.nodes_data[state]['avg_utilization'],
            'connections': st.session_state.nodes_data[state]['corridor_connections']
        } for state, coords in state_coordinates.items() if state in st.session_state.nodes_data
    ])
    
    # Determine node size based on selected metric
    if node_size_metric == "ZEV Population":
        nodes_df['node_size'] = nodes_df['zev_population'] / 10000  # Scale for visualization
        size_title = "ZEV Population"
    elif node_size_metric == "Charging Stations":
        nodes_df['node_size'] = nodes_df['charging_stations'] / 500  # Scale for visualization
        size_title = "Charging Stations"
    else:  # Fast Chargers
        nodes_df['node_size'] = nodes_df['fast_chargers'] / 200  # Scale for visualization
        size_title = "Fast Chargers"
    
    # Create network connections between states
    # For demonstration, we'll create logical connections based on geographical proximity
    # In a real application, these could be based on actual EV corridor data
    connections = []
    
    # Define some logical state connections based on geography
    nearby_states = {
        'California': ['Oregon', 'Washington'],
        'Oregon': ['California', 'Washington'],
        'Washington': ['Oregon'],
        'New York': ['Massachusetts', 'New Jersey', 'Pennsylvania'],
        'Massachusetts': ['New York'],
        'New Jersey': ['New York', 'Pennsylvania', 'Maryland'],
        'Pennsylvania': ['New York', 'New Jersey', 'Ohio', 'Maryland'],
        'Maryland': ['New Jersey', 'Pennsylvania'],
        'Texas': ['Colorado'],
        'Colorado': ['Texas'],
        'Florida': ['Georgia'],
        'Georgia': ['Florida'],
        'Ohio': ['Pennsylvania', 'Michigan'],
        'Michigan': ['Ohio', 'Illinois'],
        'Illinois': ['Michigan']
    }
    
    # Create connections with random strength
    for state1, neighbors in nearby_states.items():
        for state2 in neighbors:
            # Only include the connection once (not duplicated in reverse)
            if state1 < state2 and state1 in nodes_df['state'].values and state2 in nodes_df['state'].values:  
                # Calculate a connection strength based on ZEV adoption similarity
                state1_data = nodes_df[nodes_df['state'] == state1].iloc[0]
                state2_data = nodes_df[nodes_df['state'] == state2].iloc[0]
                
                # Similarity metric based on relative ZEV adoption and infrastructure
                zev_ratio = min(state1_data['zev_population'], state2_data['zev_population']) / \
                           max(state1_data['zev_population'], state2_data['zev_population'])
                
                charger_ratio = min(state1_data['charging_stations'], state2_data['charging_stations']) / \
                               max(state1_data['charging_stations'], state2_data['charging_stations'])
                
                connection_strength = (zev_ratio + charger_ratio) / 2
                
                # Only include connections above threshold
                if connection_strength >= connection_threshold:
                    connections.append({
                        'state1': state1,
                        'state2': state2,
                        'strength': connection_strength
                    })
    
    # Create the map figure
    fig = go.Figure()
    
    # Add connections as lines
    for conn in connections:
        state1_data = nodes_df[nodes_df['state'] == conn['state1']].iloc[0]
        state2_data = nodes_df[nodes_df['state'] == conn['state2']].iloc[0]
        
        # Line width based on connection strength
        line_width = conn['strength'] * 5
        
        fig.add_trace(go.Scattergeo(
            lon=[state1_data['lon'], state2_data['lon']],
            lat=[state1_data['lat'], state2_data['lat']],
            mode='lines',
            line=dict(width=line_width, color='rgba(0, 100, 150, 0.5)'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add nodes for each state
    fig.add_trace(go.Scattergeo(
        lon=nodes_df['lon'],
        lat=nodes_df['lat'],
        text=nodes_df['state'],
        mode='markers' + ('+text' if show_labels else ''),
        marker=dict(
            size=nodes_df['node_size'],
            color=nodes_df['utilization'],
            colorscale='Viridis',
            cmin=0.3,
            cmax=0.8,
            colorbar_title='Charger Utilization',
            line_width=0
        ),
        hovertemplate=(
            '<b>%{text}</b><br>' +
            'ZEV Population: %{customdata[0]:,}<br>' +
            'Charging Stations: %{customdata[1]:,}<br>' +
            'Fast Chargers: %{customdata[2]:,}<br>' +
            'Utilization: %{marker.color:.1%}<br>'
        ) if show_metrics else '<b>%{text}</b>',
        customdata=nodes_df[['zev_population', 'charging_stations', 'fast_chargers']],
        textfont=dict(color='black', size=10)
    ))
    
    # Update map layout
    fig.update_layout(
        title=f'ZEV Infrastructure Network - Node Size: {size_title}',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(240, 240, 240)',
            countrycolor='rgb(204, 204, 204)',
            subunitcolor='rgb(217, 217, 217)',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Display the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Add analysis and insights
    st.subheader("Network Analysis")
    
    # Calculate network metrics
    total_connections = len(connections)
    avg_connection_strength = sum(c['strength'] for c in connections) / total_connections if total_connections > 0 else 0
    most_connected_state = nodes_df.sort_values('connections', ascending=False).iloc[0]['state'] if not nodes_df.empty else "N/A"
    strongest_connection = max(connections, key=lambda x: x['strength']) if connections else None
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Corridor Connections", f"{total_connections}")
        st.metric("Average Connection Strength", f"{avg_connection_strength:.2f}")
    
    with col2:
        st.metric("Most Connected State", most_connected_state)
        if strongest_connection:
            st.metric(
                "Strongest Connection", 
                f"{strongest_connection['state1']} - {strongest_connection['state2']}"
            )
    
    # Recommendations based on network analysis
    st.subheader("Infrastructure Recommendations")
    
    # Find states with high ZEV population but low charging stations
    nodes_df['charger_per_zev'] = nodes_df['charging_stations'] / nodes_df['zev_population']
    infrastructure_gaps = nodes_df.sort_values('charger_per_zev').head(3)['state'].tolist()
    
    # Find isolated states (low connections)
    isolated_states = nodes_df.sort_values('connections').head(3)['state'].tolist()
    
    st.markdown(f"""
    Based on the network analysis, we recommend:
    
    1. **Infrastructure Investment**: Focus on expanding charging infrastructure in {', '.join(infrastructure_gaps)}
       where there is high ZEV adoption but relatively low charging station density.
       
    2. **Corridor Development**: Develop interstate EV corridors connecting {', '.join(isolated_states)}
       to improve regional connectivity and reduce range anxiety for long-distance travel.
       
    3. **Fast Charger Deployment**: Increase the percentage of fast chargers in high-utilization states
       to reduce charging times and improve user experience.
    """)

# Create tabs for different dashboard sections
tab1, tab2, tab3, tab4 = st.tabs(["U.S. Map View", "Node Network Map", "State Comparison", "Scenario Analysis"])

# Tab 1: U.S. Map View (Choropleth)
with tab1:
    st.header("ZEV Adoption by State")
    
    # Timeframe selection for projection
    projection_years = st.slider("Projection Years", 1, 30, 10)
    
    # Create a selection for map metric
    map_metric = st.selectbox(
        "Select Map Metric",
        ["ZEV Market Share (%)", "BEV Population", "Total CO2 Emissions Reduction (tons)", "Charging Stations"]
    )
    
    # Generate data for all states
    map_data = []
    
    for state_name, state_params in st.session_state.states_data.items():
        # Initial conditions
        x0 = [
            state_params['ICEV_initial'],
            state_params['BEV_initial'],
            state_params['PHEV_initial'],
            state_params['FCEV_initial']
        ]
        
        # Time points
        t = np.linspace(0, projection_years, projection_years * 12)  # Monthly resolution
        
        # Solve ODE system
        solution = odeint(
            state_dynamics, 
            x0, 
            t, 
            args=(state_params['incentives'], state_params['infrastructure_budget'], state_params['policy_stringency'])
        )
        
        # Extract relevant metrics for mapping
        final_year_idx = -1  # Use the final projection year
        
        total_vehicles = sum(solution[final_year_idx])
        zev_vehicles = solution[final_year_idx, 1] + solution[final_year_idx, 2] + solution[final_year_idx, 3]
        zev_share = 100 * zev_vehicles / total_vehicles
        
        # Calculate emissions
        baseline_emissions = state_params['ICEV_initial'] * 4.6  # Baseline annual emissions
        projected_emissions = calculate_emissions(solution)
        emissions_reduction = baseline_emissions - projected_emissions[final_year_idx]
        
        # Estimate charging stations growth based on BEV+PHEV population and infrastructure budget
        charging_growth_factor = 1 + (0.1 * projection_years * state_params['infrastructure_budget'])
        projected_stations = state_params['charging_stations'] * charging_growth_factor
        
        # Add state data to map dataset
        map_data.append({
            'state': state_name,
            'zev_share': zev_share,
            'bev_population': solution[final_year_idx, 1],
            'emissions_reduction': emissions_reduction,
            'charging_stations': projected_stations
        })
    
    map_df = pd.DataFrame(map_data)
    
    # Prepare data for choropleth map
    if map_metric == "ZEV Market Share (%)":
        choropleth_values = map_df['zev_share']
        color_scale = "Viridis"
        hover_template = "<b>%{customdata}</b><br>ZEV Share: %{z:.1f}%"
    elif map_metric == "BEV Population":
        choropleth_values = map_df['bev_population']
        color_scale = "Blues"
        hover_template = "<b>%{customdata}</b><br>BEV Population: %{z:,.0f}"
    elif map_metric == "Total CO2 Emissions Reduction (tons)":
        choropleth_values = map_df['emissions_reduction']
        color_scale = "Greens"
        hover_template = "<b>%{customdata}</b><br>CO2 Reduction: %{z:,.0f} tons"
    else:  # Charging Stations
        choropleth_values = map_df['charging_stations']
        color_scale = "Reds"
        hover_template = "<b>%{customdata}</b><br>Charging Stations: %{z:,.0f}"
    
    # Create U.S. map figure with Plotly
    fig = go.Figure(data=go.Choropleth(
        locations=map_df['state'],
        z=choropleth_values,
        locationmode='USA-states',
        colorscale=color_scale,
        colorbar_title=map_metric,
        customdata=map_df['state'],
        hovertemplate=hover_template
    ))
    
    fig.update_layout(
        geo_scope='usa',
        geo=dict(
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        ),
        title=f"{map_metric} after {projection_years} years",
        height=600
    )
    
    # Display the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights below the map
    st.subheader("Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Find highest ZEV adoption state
        max_zev_state = map_df.loc[map_df['zev_share'].idxmax()]['state']
        max_zev_share = map_df['zev_share'].max()
        st.metric("Highest ZEV Adoption", f"{max_zev_state}: {max_zev_share:.1f}%")
        
        # Find highest emissions reduction state
        max_emissions_state = map_df.loc[map_df['emissions_reduction'].idxmax()]['state']
        max_emissions = map_df['emissions_reduction'].max()
        st.metric("Highest Emissions Reduction", f"{max_emissions_state}: {max_emissions/1000000:.1f}M tons")
    
    with col2:
        # Total ZEVs across all states
        total_bevs = map_df['bev_population'].sum()
        st.metric("Total BEVs Across States", f"{total_bevs:,.0f}")
        
        # Average ZEV Share
        avg_zev_share = map_df['zev_share'].mean()
        st.metric("Average ZEV Market Share", f"{avg_zev_share:.1f}%")

# Tab 2: Node Network Map
with tab2:
    create_zev_node_map()

# Tab 3: State Comparison
with tab3:
    st.header("Compare State-Level Projections")
    
    # Select states to compare
    states_to_compare = st.multiselect(
        "Select States to Compare",
        list(st.session_state.states_data.keys()),
        default=["California", "New York", "Texas"]
    )
    
    # Select comparison metric
    comparison_metric = st.selectbox(
        "Select Comparison Metric",
        ["ZEV Market Share (%)", "BEV Population", "PHEV Population", "FCEV Population", "CO2 Emissions"]
    )
    
    # Generate comparison data
    if states_to_compare:
        comparison_data = []
        
        # Calculate projections for selected states
        for state_name in states_to_compare:
            state_params = st.session_state.states_data[state_name]
            
            # Initial conditions
            x0 = [
                state_params['ICEV_initial'],
                state_params['BEV_initial'],
                state_params['PHEV_initial'],
                state_params['FCEV_initial']
            ]
            
            # Time points (30 years max)
            projection_years = 30
            t = np.linspace(0, projection_years, projection_years * 12)  # Monthly resolution
            
            # Solve ODE system
            solution = odeint(
                state_dynamics,
                x0,
                t,
                args=(state_params['incentives'], state_params['infrastructure_budget'], state_params['policy_stringency'])
            )
            
            # Calculate metrics for each time point
            total_vehicles = np.sum(solution, axis=1)
            zev_share = 100 * (solution[:, 1] + solution[:, 2] + solution[:, 3]) / total_vehicles
            emissions = calculate_emissions(solution)
            
            years = np.linspace(2025, 2025 + projection_years, len(t))
            
            # Add to comparison data
            for i, year in enumerate(years):
                if i % 12 == 0:  # Annual data points for cleaner visualization
                    comparison_data.append({
                        'State': state_name,
                        'Year': int(year),
                        'ZEV Share (%)': zev_share[i],
                        'BEV Population': solution[i, 1],
                        'PHEV Population': solution[i, 2],
                        'FCEV Population': solution[i, 3],
                        'CO2 Emissions': emissions[i]
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison chart
        if comparison_metric == "ZEV Market Share (%)":
            y_column = 'ZEV Share (%)'
            title = "ZEV Market Share Projection by State"
            y_axis_title = "Market Share (%)"
        elif comparison_metric == "BEV Population":
            y_column = 'BEV Population'
            title = "BEV Population Projection by State"
            y_axis_title = "Number of Vehicles"
        elif comparison_metric == "PHEV Population":
            y_column = 'PHEV Population'
            title = "PHEV Population Projection by State"
            y_axis_title = "Number of Vehicles"
        elif comparison_metric == "FCEV Population":
            y_column = 'FCEV Population'
            title = "FCEV Population Projection by State"
            y_axis_title = "Number of Vehicles"
        else:  # CO2 Emissions
            y_column = 'CO2 Emissions'
            title = "CO2 Emissions Projection by State"
            y_axis_title = "Emissions (tons CO2)"
        
        fig = px.line(
            comparison_df, 
            x='Year', 
            y=y_column, 
            color='State',
            title=title
        )
        
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title=y_axis_title,
            legend_title="State",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add data table below
        st.subheader("Projection Data Table")
        pivot_table = comparison_df.pivot_table(
            index='Year',
            columns='State',
            values=y_column,
            aggfunc='mean'
        ).reset_index()
        
        st.dataframe(pivot_table, use_container_width=True)

# Tab 4: Scenario Analysis
with tab4:
    st.header("Policy Scenario Analysis")
    
    # Select state for scenario analysis
    scenario_state = st.selectbox(
        "Select State for Scenario Analysis",
        list(st.session_state.states_data.keys())
    )
    
    # Get current parameters for selected state
    current_params = st.session_state.states_data[scenario_state]
    
    # Create columns for parameter adjustment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario_incentives = st.slider(
            "EV Incentives ($)",
            0, 10000, int(current_params['incentives']),
            step=500
        )
    
    with col2:
        scenario_infrastructure = st.slider(
            "Infrastructure Budget ($ billions)",
            0.0, 5.0, float(current_params['infrastructure_budget']),
            step=0.1
        )
    
    with col3:
        scenario_policy = st.slider(
            "Policy Stringency (0-1)",
            0.0, 1.0, float(current_params['policy_stringency']),
            step=0.05
        )
    
    # Button to run scenario
    if st.button("Run Scenario Analysis"):
        # Set up baseline and scenario parameters
        baseline_params = (
            current_params['incentives'],
            current_params['infrastructure_budget'],
            current_params['policy_stringency']
        )
        
        scenario_params = (
            scenario_incentives,
            scenario_infrastructure,
            scenario_policy
        )
        
        # Initial conditions
        x0 = [
            current_params['ICEV_initial'],
            current_params['BEV_initial'],
            current_params['PHEV_initial'],
            current_params['FCEV_initial']
        ]
        
        # Time points (20 years)
        projection_years = 20
        t = np.linspace(0, projection_years, projection_years * 12)  # Monthly resolution
        
        # Solve ODE system for baseline
        baseline_solution = odeint(
            state_dynamics,
            x0,
            t,
            args=baseline_params
        )
        
        # Solve ODE system for scenario
        scenario_solution = odeint(
            state_dynamics,
            x0,
            t,
            args=scenario_params
        )
        
        # Calculate metrics
        years = np.linspace(2025, 2025 + projection_years, len(t))
        
        # ZEV market share
        baseline_total = np.sum(baseline_solution, axis=1)
        baseline_zev = baseline_solution[:, 1] + baseline_solution[:, 2] + baseline_solution[:, 3]
        baseline_share = 100 * baseline_zev / baseline_total
        
        scenario_total = np.sum(scenario_solution, axis=1)
        scenario_zev = scenario_solution[:, 1] + scenario_solution[:, 2] + scenario_solution[:, 3]
        scenario_share = 100 * scenario_zev / scenario_total
        
        # CO2 emissions
        baseline_emissions = calculate_emissions(baseline_solution)
        scenario_emissions = calculate_emissions(scenario_solution)
        
        # Create comparison dataframe
        scenario_comparison = []
        
        for i, year in enumerate(years):
            if i % 12 == 0:  # Annual data points
                scenario_comparison.append({
                    'Year': int(year),
                    'Scenario': 'Baseline',
                    'ZEV Share (%)': baseline_share[i],
                    'BEV Population': baseline_solution[i, 1],
                    'CO2 Emissions': baseline_emissions[i]
                })
                
                scenario_comparison.append({
                    'Year': int(year),
                    'Scenario': 'New Policy',
                    'ZEV Share (%)': scenario_share[i],
                    'BEV Population': scenario_solution[i, 1],
                    'CO2 Emissions': scenario_emissions[i]
                })
        
        scenario_df = pd.DataFrame(scenario_comparison)
        
        # Create scenario comparison charts
        st.subheader(f"ZEV Market Share Projection for {scenario_state}")
        
        fig1 = px.line(
            scenario_df,
            x='Year',
            y='ZEV Share (%)',
            color='Scenario',
            title="ZEV Market Share: Baseline vs. New Policy"
        )
        
        fig1.update_layout(
            xaxis_title="Year",
            yaxis_title="Market Share (%)",
            legend_title="Scenario",
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Emissions comparison
        st.subheader(f"CO2 Emissions Projection for {scenario_state}")
        
        fig2 = px.line(
            scenario_df,
            x='Year',
            y='CO2 Emissions',
            color='Scenario',
            title="CO2 Emissions: Baseline vs. New Policy"
        )
        
        fig2.update_layout(
            xaxis_title="Year",
            yaxis_title="CO2 Emissions (tons)",
            legend_title="Scenario",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Calculate key metrics for summary
        final_baseline_share = scenario_df[scenario_df['Scenario'] == 'Baseline']['ZEV Share (%)'].iloc[-1]
        final_scenario_share = scenario_df[scenario_df['Scenario'] == 'New Policy']['ZEV Share (%)'].iloc[-1]
        share_difference = final_scenario_share - final_baseline_share
        
        baseline_emissions_2045 = scenario_df[scenario_df['Scenario'] == 'Baseline']['CO2 Emissions'].iloc[-1]
        scenario_emissions_2045 = scenario_df[scenario_df['Scenario'] == 'New Policy']['CO2 Emissions'].iloc[-1]
        emissions_reduction = baseline_emissions_2045 - scenario_emissions_2045
        
        # Cost-effectiveness calculation
        incentive_change = scenario_incentives - current_params['incentives']
        infrastructure_change = (scenario_infrastructure - current_params['infrastructure_budget']) * 1000000000  # Convert to dollars
        
        # Estimate total vehicles affected over period
        avg_annual_new_vehicles = current_params['ICEV_initial'] * 0.07  # Assume 7% annual replacement rate
        total_affected_vehicles = avg_annual_new_vehicles * projection_years
        
        # Total incentive cost
        total_incentive_cost = total_affected_vehicles * incentive_change * scenario_share.mean() / 100
        
        # Total policy cost
        total_policy_cost = total_incentive_cost + infrastructure_change
        
        # Cost per ton of CO2 reduced (if there's a reduction)
        if emissions_reduction > 0:
            cumulative_emissions_reduction = sum(baseline_emissions - scenario_emissions)
            cost_per_ton = total_policy_cost / cumulative_emissions_reduction
        else:
            cost_per_ton = 0
        
        # Display cost-effectiveness metrics
        st.subheader("Policy Cost-Effectiveness Analysis")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "ZEV Share Increase by 2045",
                f"{share_difference:.1f}%",
                delta=f"{share_difference:.1f}%"
            )
        
        with metric_col2:
            st.metric(
                "Emissions Reduction by 2045",
                f"{emissions_reduction/1000:.1f}K tons",
                delta=f"{-emissions_reduction/1000:.1f}K tons" if emissions_reduction > 0 else "0 tons"
            )
        
        with metric_col3:
            if emissions_reduction > 0:
                st.metric(
                    "Cost per Ton CO2 Reduced",
                    f"${cost_per_ton:.2f}",
                    delta=None
                )
            else:
                st.metric(
                    "Cost per Ton CO2 Reduced",
                    "N/A (No Reduction)",
                    delta=None
                )
        
        # Policy recommendations
        st.subheader("Policy Recommendations")
        
        if share_difference > 5 and cost_per_ton < 100:
            st.success("""
                The proposed policy scenario shows strong cost-effectiveness with significant ZEV adoption increases.
                Consider implementing this policy package as designed.
            """)
        elif share_difference > 2 and cost_per_ton < 200:
            st.info("""
                The proposed policy shows moderate effectiveness. Consider adjusting the incentive levels
                to improve cost-effectiveness while maintaining adoption targets.
            """)
        else:
            st.warning("""
                The proposed policy shows limited effectiveness relative to its cost.
                Consider focusing more on infrastructure development or targeting incentives to specific vehicle segments.
            """)

# Add a methodology section at the bottom to explain the model
st.markdown("---")
st.subheader("Methodology: State-Space Modeling for ZEV Adoption")
with st.expander("View Methodology Details"):
    st.markdown("""
    ### Overview
    To extend our ZEV adoption analysis from California to a nationwide scale, we adopt a modular state-space modeling approach. 
    Due to data limitations, our initial implementation includes 15 states, each modeled independently. This enables us to 
    incorporate state-specific characteristics such as varying incentive structures, infrastructure investments, 
    demographic trends, and policy targets.
    
    ### State-Specific Dynamical Model
    For each state, we define a dynamical system:
    
    ```
    x(t) = f(x(t)) + g(Incentives, Infrastructure, Policy)
    ```
    
    Where:
    - `x(t)`: State-specific vector of system variables, e.g., ICEV, BEV, PHEV, FCEV populations, VMT, CO₂ emissions, and charging infrastructure.
    - `f(·)`: Autonomous dynamics, including growth, decay, and transition flows.
    - `g(·)`: External forcing functions capturing the impact of state-specific incentives, infrastructure budgets, and policy changes.
    
    ### Geospatial Integration
    To support intuitive visualization and policy relevance, we integrate each state's model into a dashboard environment 
    using a geospatial map of the United States. Each state can be selected to display:
    
    - Time-series trajectories of key variables (e.g., ZEV population, emissions)
    - Scenario comparisons based on changes in incentive strategies
    - Cost-effectiveness of policy interventions
    
    ### Network Analysis
    The node-based geographical map provides insights into regional adoption patterns and infrastructure corridors. 
    It enables the identification of:
    
    - States with infrastructure gaps
    - Potential interstate corridors for charging station deployment
    - Regional adoption clusters
    
    ### Future Expansion
    The framework is designed for future inclusion of:
    
    - Additional states as data quality improves
    - Federally coordinated scenarios (e.g., national-level budget allocation)
    - Machine learning models for predictive inference and data imputation
    """)

# Footer with information
st.markdown("---")
st.caption("Nationwide ZEV Adoption Modeling Dashboard | Developed for effective presentation of state-space modeling approach")