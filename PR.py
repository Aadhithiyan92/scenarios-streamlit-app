import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(layout="wide", page_title="Semiconductor Supply Chain Analysis"

# Title and Introduction
st.title("Environmental-Economic Sensitivity Analysis of US Regional Semiconductor Supply Chains")

# Sidebar for Navigation
page = st.sidebar.selectbox(
    "Select Section",
    ["Overview", "Objectives", "Methodology", "Data Requirements", "Regional Analysis", "Results"]
)

if page == "Overview":
    st.header("Research Overview")
    st.write("""
    This research introduces a novel nonlinear dynamical systems framework for analyzing 
    the complex interactions between environmental sustainability factors and semiconductor 
    supply chain resilience across key US manufacturing regions.
    """)
    
    # Key Features in columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Innovations")
        st.write("""
        - Nonlinear coupling of environmental-economic factors
        - Regional sensitivity analysis framework
        - Critical threshold identification methods
        - Advanced stability analysis techniques
        """)
    
    with col2:
        st.subheader("Expected Impact")
        st.write("""
        - Enhanced supply chain resilience strategies
        - Regional policy recommendations
        - Sustainability-oriented manufacturing practices
        - Improved risk assessment frameworks
        """)

elif page == "Objectives":
    st.header("Research Objectives")
    
    st.subheader("Primary Objectives")
    objectives = {
        "System Modeling": "Develop a nonlinear dynamical system incorporating environmental-economic coupling",
        "Regional Analysis": "Analyze sensitivity across different US manufacturing regions",
        "Stability Assessment": "Identify critical thresholds and stability boundaries",
        "Policy Impact": "Develop strategic recommendations for regional resilience"
    }
    
    for title, description in objectives.items():
        with st.expander(title):
            st.write(description)

elif page == "Methodology":
    
    
    st.header("Analysis Methods")
    methods = {
        "Stability Analysis": "Lyapunov stability analysis near equilibrium points",
        "Bifurcation Analysis": "Identification of critical parameter thresholds",
        "Sensitivity Analysis": "Parameter impact assessment",
        "Regional Comparison": "Comparative analysis across different US regions"
    }
    
    for method, description in methods.items():
        st.markdown(f"**{method}:** {description}")

elif page == "Data Requirements":
    st.header("Data Requirements")
    
    # Create tabs for different data categories
    tab1, tab2, tab3 = st.tabs(["Easily Accessible", "Moderately Difficult", "Challenging"])
    
    with tab1:
        st.subheader("Easily Accessible Data")
        easy_data = {
            "Source": ["DOE", "State Agencies", "EPA"],
            "Data Type": ["Regional Energy Consumption", "Water Usage Permits", "Environmental Compliance"],
            "Update Frequency": ["Monthly", "Quarterly", "Annual"]
        }
        st.table(pd.DataFrame(easy_data))
    
    with tab2:
        st.subheader("Moderately Difficult Data")
        mod_data = {
            "Source": ["Company Reports", "Industry Surveys", "Trade Associations"],
            "Data Type": ["Facility Energy Usage", "Water Recycling Rates", "Production Capacity"],
            "Access Method": ["Public Reports", "Paid Subscriptions", "Membership"]
        }
        st.table(pd.DataFrame(mod_data))
    
    with tab3:
        st.subheader("Challenging Data")
        challenging_data = {
            "Source": ["Internal Records", "Proprietary Data", "Corporate Plans"],
            "Data Type": ["Efficiency Metrics", "Production Costs", "Environmental Targets"],
            "Challenges": ["Confidential", "Limited Access", "Non-standardized"]
        }
        st.table(pd.DataFrame(challenging_data))

elif page == "Regional Analysis":
    st.header("Regional Analysis")
    
    # Regional comparison using columns
    regions = {
        "Southwest (AZ, NM)": {
            "Characteristics": ["Water scarcity challenges", "High solar potential", "Major players: Intel, TSMC"],
            "Key Concerns": ["Water availability", "Cooling efficiency", "Energy costs"],
            "Opportunities": ["Solar power integration", "Advanced recycling", "New fab developments"]
        },
        "Pacific Northwest (OR, WA)": {
            "Characteristics": ["Hydroelectric power", "Stable water supply", "Major player: Intel"],
            "Key Concerns": ["Environmental regulations", "Natural disasters", "Grid reliability"],
            "Opportunities": ["Green energy expansion", "Water conservation", "Sustainable practices"]
        },
        "Texas": {
            "Characteristics": ["Independent power grid", "Mixed energy sources", "Major players: Samsung, TI"],
            "Key Concerns": ["Grid stability", "Water rights", "Climate impacts"],
            "Opportunities": ["Energy diversity", "Manufacturing expansion", "Technology innovation"]
        },
        "Northeast (NY)": {
            "Characteristics": ["Stable resources", "Strict regulations", "Major player: GlobalFoundries"],
            "Key Concerns": ["Regulatory compliance", "Operating costs", "Winter impacts"],
            "Opportunities": ["Research collaboration", "Policy leadership", "Workforce development"]
        }
    }
    
    for region, details in regions.items():
        with st.expander(region):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Characteristics**")
                for item in details["Characteristics"]:
                    st.write(f"• {item}")
            with col2:
                st.markdown("**Key Concerns**")
                for item in details["Key Concerns"]:
                    st.write(f"• {item}")
            with col3:
                st.markdown("**Opportunities**")
                for item in details["Opportunities"]:
                    st.write(f"• {item}")

elif page == "Results":
    st.header("Expected Results")
    
    # Create three columns for different result categories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Research Outcomes")
        outcomes = [
            "Regional Stability Maps",
            "Sensitivity Metrics",
            "Risk Assessment Framework",
            "Parameter Thresholds"
        ]
        for outcome in outcomes:
            st.write(f"• {outcome}")
    
    with col2:
        st.subheader("Expected Impact")
        impacts = [
            "Policy Recommendations",
            "Industry Guidelines",
            "Resilience Strategies",
            "Best Practices"
        ]
        for impact in impacts:
            st.write(f"• {impact}")
    
    with col3:
        st.subheader("Future Applications")
        applications = [
            "Model Extension",
            "Decision Support",
            "Risk Management",
            "Strategic Planning"
        ]
        for application in applications:
            st.write(f"• {application}")
