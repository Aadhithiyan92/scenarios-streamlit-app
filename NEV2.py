import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta

# This can be integrated with the main application or used as a standalone component

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
    
    # State centroids (approximate lat/long coordinates for state centers)
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
    
    # Sample data for demonstration
    if 'nodes_data' not in st.session_state:
        # Generate sample data for ZEV adoption and charging infrastructure
        st.session_state.nodes_data = {
            state: {
                'zev_population': np.random.randint(10000, 1000000),
                'charging_stations': np.random.randint(1000, 50000),
                'avg_utilization': np.random.uniform(0.3, 0.8),
                'fast_chargers_pct': np.random.uniform(0.1, 0.4),
                'corridor_connections': np.random.randint(1, 5)
            } for state in state_coordinates.keys()
        }
    
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
        } for state, coords in state_coordinates.items()
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
            if state1 < state2:  
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
    most_connected_state = nodes_df.sort_values('connections', ascending=False).iloc[0]['state']
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

# Function to be called from the main app
def add_nodes_map_to_app():
    create_zev_node_map()

# If run standalone
if __name__ == "__main__":
    st.set_page_config(
        page_title="ZEV Network Map",
        page_icon="🚗",
        layout="wide"
    )
    create_zev_node_map()