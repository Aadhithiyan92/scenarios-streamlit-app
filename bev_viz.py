import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout="wide", page_title="BEV Adoption Scenarios Comparison")

# Create the combined data for the scenarios
data = pd.DataFrame({
    'Scenarios': ['Base Scenario', 'Infrastructure Heavy', 'Incentive Heavy', 'Balanced', 'High Incentive', 'High Growth', 'Combined'],
    'Type': ['Budget Allocation', 'Budget Allocation', 'Budget Allocation', 'Budget Allocation', 
             'Growth & Incentive', 'Growth & Incentive', 'Growth & Incentive'],
    'Vehicles': [2.4, 4.4, 3.2, 3.9, 4.6, 5.6, 6.1]
})

# Create separate dataframes for each scenario with normalized percentages
budget_data = pd.DataFrame({
    'Scenario': ['Base Scenario', 'Infrastructure Heavy', 'Incentive Heavy', 'Balanced'],
    'Vehicles': [2.4, 4.4, 3.2, 3.9],
    'Improvement': [0, 91.5, 30.4, 52.8]  # Normalized from 182.9%, 60.9%, 105.6%
})

growth_data = pd.DataFrame({
    'Scenario': ['Base Scenario', 'High Incentive', 'High Growth', 'Combined'],
    'Vehicles': [4.2, 4.6, 5.6, 6.1],
    'Improvement': [0, 10.3, 33.3, 45.2]
})

# Title and description
st.title("BEV Adoption Scenarios Comparison (2023-2027)")
st.write("Compare Budget Allocation vs Growth & Incentive Scenarios")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Comparison View", "Individual Scenarios", "Key Insights"])

with tab1:
    st.header("Scenario Comparison")
    
    # Create the bar chart
    fig = go.Figure()
    
    # Add bars for Budget Allocation
    budget_mask = data['Type'] == 'Budget Allocation'
    fig.add_trace(go.Bar(
        x=data[budget_mask]['Scenarios'],
        y=data[budget_mask]['Vehicles'],
        name='Budget Allocation',
        marker_color='rgb(66, 133, 244)',
        width=0.4
    ))
    
    # Add bars for Growth & Incentive
    growth_mask = data['Type'] == 'Growth & Incentive'
    fig.add_trace(go.Bar(
        x=data[growth_mask]['Scenarios'],
        y=data[growth_mask]['Vehicles'],
        name='Growth & Incentive',
        marker_color='rgb(15, 157, 88)',
        width=0.4
    ))
    
    # Update layout
    fig.update_layout(
        title='Vehicle Adoption by Scenario Type',
        xaxis_title='Scenarios',
        yaxis_title='Million Vehicles',
        yaxis=dict(
            range=[0, 7],
            dtick=1,
            gridcolor='lightgrey'
        ),
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500,
        margin=dict(t=100)
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Individual Scenarios Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Budget Allocation Scenario")
        
        # Create visualization for Budget Allocation
        fig_budget = go.Figure()
        fig_budget.add_trace(go.Bar(
            x=budget_data['Scenario'],
            y=budget_data['Vehicles'],
            marker_color='rgb(66, 133, 244)',
            width=0.6,
            text=budget_data['Improvement'].apply(lambda x: f'+{x}%' if x > 0 else '0%'),
            textposition='outside'
        ))
        
        fig_budget.update_layout(
            title='Budget Allocation Scenario Outcomes',
            yaxis=dict(
                title='Million Vehicles',
                range=[0, 5],
                dtick=1,
                gridcolor='lightgrey'
            ),
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        
        fig_budget.update_xaxes(showgrid=False)
        fig_budget.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        st.plotly_chart(fig_budget, use_container_width=True)
        
        st.write("""
        - Base: 2.4M vehicles
        - Infrastructure Heavy: 4.4M vehicles (91.5% improvement)
        - Incentive Heavy: 3.2M vehicles (30.4% improvement)
        - Balanced: 3.9M vehicles (52.8% improvement)
        """)
    
    with col2:
        st.subheader("Growth & Incentive Scenario")
        
        # Create visualization for Growth & Incentive
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Bar(
            x=growth_data['Scenario'],
            y=growth_data['Vehicles'],
            marker_color='rgb(15, 157, 88)',
            width=0.6,
            text=growth_data['Improvement'].apply(lambda x: f'+{x}%' if x > 0 else '0%'),
            textposition='outside'
        ))
        
        fig_growth.update_layout(
            title='Growth & Incentive Scenario Outcomes',
            yaxis=dict(
                title='Million Vehicles',
                range=[0, 7],
                dtick=1,
                gridcolor='lightgrey'
            ),
            plot_bgcolor='white',
            height=400,
            showlegend=False
        )
        
        fig_growth.update_xaxes(showgrid=False)
        fig_growth.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        st.plotly_chart(fig_growth, use_container_width=True)
        
        st.write("""
        - Base: 4.2M vehicles
        - High Incentive: 4.6M vehicles (10.3% improvement)
        - High Growth: 5.6M vehicles (33.3% improvement)
        - Combined: 6.1M vehicles (45.2% improvement)
        """)

with tab3:
    st.header("Key Insights")
    
    st.markdown("""
    #### Budget Allocation Strategy
    - Infrastructure-heavy approach shows strongest growth (91.5% improvement)
    - Balanced approach provides stable growth path (52.8% improvement)
    - Base scenario represents conservative projection (2.4M vehicles)
    
    #### Growth & Incentive Strategy
    - Combined approach achieves highest overall adoption (45.2% improvement)
    - High Growth scenario demonstrates significant potential (33.3% improvement)
    - High Incentive shows moderate improvement (10.3% improvement)
    
    #### Comparative Analysis
    - Growth & Incentive strategies generally achieve higher absolute adoption rates
    - Infrastructure investment crucial for initial market development
    - Combined approaches in both strategies show superior results
    """)

# Add sidebar
with st.sidebar:
    st.header("About")
    st.write("""
    This visualization compares two different approaches to BEV adoption:
    1. Budget Allocation Strategy
    2. Growth & Incentive Strategy
    
    Use the tabs above to explore different aspects of the analysis.
    
    Note: Improvement percentages are normalized relative to their respective base scenarios.
    """)