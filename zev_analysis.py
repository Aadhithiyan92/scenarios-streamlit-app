import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Current Investment Distributions and Future Recommendations",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean CSS without boxes and shadows
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 0;
            max-width: 100%;
            background-color: #ffffff;
        }
        
        .block-container {
            padding: 1rem 2rem;
            max-width: 100%;
        }
        
        /* Header styling */
        h1 {
            color: #1e293b;
            font-size: 2rem;
            font-weight: 600;
            text-align: center;
            margin: 1rem 0;
        }
        
        /* Section headers */
        h2 {
            color: #334155;
            font-size: 1.5rem;
            font-weight: 500;
            margin: 1rem 0;
        }
        
        /* Remove default Streamlit padding and margins */
        .element-container {
            margin: 0;
            padding: 0;
        }
        
        /* Clean metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem;
            font-weight: 500;
        }
        
        /* Remove default Streamlit borders and shadows */
        .css-1d391kg, .css-12oz5g7 {
            border: none;
            box-shadow: none;
        }
        
        /* Clean tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        
        /* Remove plot container styling */
        .plot-container {
            margin: 0;
            padding: 0;
        }
    </style>
""", unsafe_allow_html=True)

# Data structures
timeline_data = pd.DataFrame({
    "Program": ["CVRP", "CVRP", "CC4A", "CC4A", "DCAP", "DCAP", "CVAP", "CVAP"],
    "Year": [2010, 2024, 2015, 2024, 2016, 2024, 2018, 2024],
    "Type": ["Start", "End", "Start", "End", "Start", "End", "Start", "End"],
    "Amount_M": [1511.9, 1511.9, 174.4, 174.4, 2.9, 2.9, 25.4, 25.4]
})

programs_info = {
    "CVRP": {
        "Start": 2010,
        "Total Investment": "1511.9",
        "Status": "Active",
        "Description": "Clean Vehicle Rebate Project - Direct consumer rebates"
    },
    "CC4A": {
        "Start": 2015,
        "Total Investment": "174.4",
        "Status": "Active",
        "Description": "Clean Cars 4 All - Vehicle replacement program"
    },
    "DCAP": {
        "Start": 2016,
        "Total Investment": "2.9",
        "Status": "Active",
        "Description": "Drive Clean Assistance Program"
    },
    "CVAP": {
        "Start": 2018,
        "Total Investment": "25.4",
        "Status": "Active",
        "Description": "Clean Vehicle Assistance Program"
    }
}

infrastructure_data = pd.DataFrame({
    "Program": ["CALeVIP", "EnergIIZE", "Clean Transportation Program"],
    "Amount_M": [287, 50, 100],
    "Description": [
        "California Electric Vehicle Infrastructure Project",
        "Energy Infrastructure Incentives for Zero-Emission Commercial Vehicles",
        "Annual funding for clean transportation infrastructure"
    ]
})

def create_timeline():
    fig = go.Figure()
    
    programs = ["CVRP", "CC4A", "DCAP", "CVAP"]
    colors = {
        "CVRP": '#38B2AC',
        "CC4A": '#4299E1',
        "DCAP": '#48BB78',
        "CVAP": '#2D3748'
    }
    
    for prog in programs:
        prog_data = timeline_data[timeline_data['Program'] == prog]
        start_year = prog_data['Year'].iloc[0]
        amount = prog_data['Amount_M'].iloc[0]

        # Add bar
        fig.add_trace(go.Scatter(
            x=[start_year, 2024],
            y=[prog, prog],
            mode='lines',
            line=dict(color=colors[prog], width=25),
            name=prog,
            hovertemplate=f"<b>{prog}</b><br>Start: {start_year}<br>Investment: ${amount:,.1f}M<extra></extra>"
        ))
        
        # Add text label at the end
        fig.add_trace(go.Scatter(
            x=[2024],
            y=[prog],
            mode='text',
            text=[f'${amount:,.1f}M'],
            textposition='middle right',
            textfont=dict(size=12, color='black'),
            showlegend=False
        ))

        # Add start point diamond
        fig.add_trace(go.Scatter(
            x=[start_year],
            y=[prog],
            mode='markers',
            marker=dict(
                symbol='diamond',
                size=12,
                color='white',
                line=dict(color=colors[prog], width=2)
            ),
            showlegend=False,
            hovertemplate=f"Program Start: {start_year}<extra></extra>"
        ))

    fig.update_layout(
        title={
            'text': "California ZEV Program Timeline (2010-Present)",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis=dict(
            title="Year",
            range=[2009, 2025],
            tickmode='array',
            ticktext=[str(year) for year in range(2010, 2025, 2)],
            tickvals=list(range(2010, 2025, 2)),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            title="Program",
            showgrid=False,
            categoryorder='array',
            categoryarray=programs
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_investment_comparison():
    # Calculate totals
    incentive_total = sum([float(info['Total Investment']) for info in programs_info.values()])
    infrastructure_total = infrastructure_data['Amount_M'].sum()
    
    # Create DataFrame
    comparison_data = pd.DataFrame({
        'Category': ['Incentives', 'Growth'],
        'Amount_M': [incentive_total, infrastructure_total]
    })
    
    comparison_data['Percentage'] = (comparison_data['Amount_M'] / comparison_data['Amount_M'].sum()) * 100
    
    fig = px.pie(comparison_data, 
                 values='Amount_M', 
                 names='Category',
                 title='Current Investment Distribution',
                 color_discrete_sequence=['#3498db', '#2ecc71'])
    
    fig.update_traces(
        textinfo='value+percent',
        texttemplate='%{value:,.1f}<br>%{percent:.1f}%',
        hovertemplate="<b>%{label}</b><br>" +
                     "$%{value:.1f}M<br>" +
                     "%{percent:.1f}%<extra></extra>"
    )
    
    fig.update_layout(
        title={
            'text': "Current Investment Distribution Upto 2023",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16)
        },
        width=600,
        height=400,
        margin=dict(l=80, r=80, t=100, b=80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def create_program_breakdowns():
    # Create incentive breakdown
    incentive_data = pd.DataFrame({
        'Program': ['CVRP', 'CC4A', 'CVAP', 'DCAP'],
        'Amount': [1511.9, 174.4, 25.4, 2.9],
        'Category': ['Incentives'] * 4
    })
    
    # Create infrastructure breakdown
    infrastructure_breakdown = pd.DataFrame({
        'Program': ['CALeVIP', 'EnergIIZE', 'Clean Transportation Program'],
        'Amount': [287.0, 50.0, 100.0],
        'Category': ['Growth'] * 3
    })
    
    # Calculate percentages for each category separately
    incentive_total = incentive_data['Amount'].sum()
    infrastructure_total = infrastructure_breakdown['Amount'].sum()
    
    incentive_data['Percentage'] = (incentive_data['Amount'] / incentive_total) * 100
    infrastructure_breakdown['Percentage'] = (infrastructure_breakdown['Amount'] / infrastructure_total) * 100
    
    # Create incentive breakdown figure
    fig_incentives = go.Figure()
    fig_incentives.add_trace(go.Bar(
        y=incentive_data['Program'],
        x=incentive_data['Percentage'],
        orientation='h',
        text=[f"${amount:.1f}M ({pct:.1f}%)" 
              for amount, pct in zip(incentive_data['Amount'], incentive_data['Percentage'])],
        textposition='auto',
        marker_color=['#38B2AC', '#4299E1', '#48BB78', '#2D3748'],
        name='Incentives'
    ))
    
    fig_incentives.update_layout(
        title="ZEV Incentive Program Distribution",
        xaxis_title="Percentage of Total Incentive Funding",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Create infrastructure breakdown figure
    fig_infrastructure = go.Figure()
    fig_infrastructure.add_trace(go.Bar(
        y=infrastructure_breakdown['Program'],
        x=infrastructure_breakdown['Percentage'],
        orientation='h',
        text=[f"${amount:.1f}M ({pct:.1f}%)" 
              for amount, pct in zip(infrastructure_breakdown['Amount'], infrastructure_breakdown['Percentage'])],
        textposition='auto',
        marker_color=['#2ecc71', '#27ae60', '#16a085'],
        name='Infrastructure'
    ))
    
    fig_infrastructure.update_layout(
        title="Infrastructure Investment Distribution",
        xaxis_title="Percentage of Total Infrastructure Funding",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig_incentives, fig_infrastructure

def create_recommended_distribution():
    # Create DataFrame for recommended distribution
    recommended_data = pd.DataFrame({
        'Category': ['Growth-Focused', 'Incentives'],
        'Percentage': [70, 30]
    })
    
    fig = px.pie(
        recommended_data, 
        values='Percentage',
        names='Category',
        title='Recommended Distribution (Based on Sensitivity)',
        color_discrete_sequence=['#2ecc71', '#3498db']  # Green for Growth, Blue for Incentives
    )
    
    # Use built-in percentage display
    fig.update_traces(
        textinfo='percent',
        hovertemplate="<b>%{label}</b><br>%{percent:.0f}%<extra></extra>"
    )
    
    fig.update_layout(
        title={
            'text': "Recommended Distribution (Based on Sensitivity)",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16)
        },
        width=600,
        height=400,
        margin=dict(l=80, r=80, t=100, b=80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def main():
    # Title section
    st.title('Current Investment Distributions and Future Recommendations')
    
    # Create two tabs
    tab1, tab2 = st.tabs(["Timeline & Distribution", "Recommendations"])
    
    with tab1:
        # Timeline section
        st.header('Program Timeline')
        timeline_fig = create_timeline()
        st.plotly_chart(timeline_fig, use_container_width=True, config={'displayModeBar': False})
        
        # Investment Distribution section
        st.header('Investment Distribution Analysis')
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_dist = create_investment_comparison()
            st.plotly_chart(current_dist, use_container_width=True, config={'displayModeBar': False})
        
        # Program Breakdowns section (moved below current investment distribution)
        st.header('Detailed Program Breakdowns')
        breakdown_col1, breakdown_col2 = st.columns([1, 1])
        
        incentive_fig, infrastructure_fig = create_program_breakdowns()
        
        with breakdown_col1:
            st.subheader('Incentive Programs')
            st.plotly_chart(incentive_fig, use_container_width=True, config={'displayModeBar': False})
            
        with breakdown_col2:
            st.subheader('Infrastructure Programs')
            st.plotly_chart(infrastructure_fig, use_container_width=True, config={'displayModeBar': False})
    
    with tab2:
        # Recommendations section
        st.markdown("## Recommendations")
        
        # Add the recommended distribution pie chart
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            recommended_dist = create_recommended_distribution()
            st.plotly_chart(recommended_dist, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
            ### Strategic Priorities
            
            1. **Growth-Focused Initiatives (70% of budget)**
               - Infrastructure Development (40%)
                 - Expand charging network coverage
                 - R&D for advanced technologies
                 - Enhance grid capacity in key corridors
                 - Develop fast-charging hubs
                 - Educational Campaigns
               - Manufacturing Support (30%)
                 - Supply chain development
                 - Production capacity expansion
                 - Technology innovation support
            
            2. **Consumer Incentives (30% of budget)**
               - Targeted rebate programs
               - Income-based subsidies
               - Fleet transition support
        """)
        
        st.markdown("### Additional Insights")
        metric_col1, metric_col2 = st.columns([1, 1])
        
        with metric_col1:
            st.metric(
                "Growth Parameter Sensitivity",
                "51.0%",
                "High impact on adoption rates",
                help="Model sensitivity coefficient showing strong influence of growth parameters"
            )
            st.markdown("""
                #### Key Finding
                The growth parameter shows significant influence on BEV adoption, suggesting that 
                infrastructure and growth initiatives have substantial leverage in accelerating adoption.
            """)
        
        with metric_col2:
            st.metric(
                "Recommended Growth Allocation",
                "70%",
                "vs current 20.3%",
                help="Optimal allocation based on sensitivity analysis"
            )
            st.markdown("""
                #### Recommendation
                Shift investment focus toward growth initiatives while maintaining targeted incentive 
                programs for maximum effectiveness.
            """)
        
        st.markdown("""
        ### Current Challenges
        - **Heavy Incentive Focus**: 79.7% of funds in direct incentives
        - **Dominated by CVRP**: 88.2% of incentive funds
        - **Limited Growth Investment**: Only 20.3% for infrastructure

        ### Proposed Changes
        - **Rebalance Portfolio**: Shift to 70% growth focus
        - **Optimize Incentives**: Streamline to 30% of funding
        - **Prioritize Infrastructure**: Expand charging network
        - **Enhance Manufacturing**: Support supply chain growth
        """)

        st.markdown("""
        <div class="data-sources">
            <h4>Data Sources</h4>
            <ul>
                <li>🏢 California Air Resources Board (CARB)</li>
                <li>⚡ California Energy Commission (CEC)</li>
                <li>🚗 Clean Transportation Program Investment Plan</li>
                <li>📊 Model Sensitivity Analysis Results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
