import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="TSMC Environmental-Economic Impact Research",
    page_icon="🔬",
    layout="wide"
)

# Create sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/TSMC_Logo.svg/1200px-TSMC_Logo.svg.png", width=200)
st.sidebar.title("Research Navigation")
page = st.sidebar.radio("Select Page", ["Project Overview", "Data Availability", "Methodology Approach", "Model Validation Plan", "Expected Outcomes"])

# Create header
st.title("Environmental-Economic Impact Analysis of Semiconductor Supply Chain")
st.subheader("Case Study: TSMC Arizona")

# Project Overview page
if page == "Project Overview":
    st.header("Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Research Focus
        This project investigates the environmental-economic impacts of semiconductor 
        supply chains with a focus on resilience metrics. We've selected TSMC's 
        Arizona facility as our primary case study to understand these dynamics 
        in the context of US semiconductor manufacturing expansion.
        
        ### Research Objectives
        - Quantify environmental impacts of semiconductor manufacturing
        - Assess economic benefits at regional and national levels
        - Develop resilience metrics for semiconductor supply chains
        - Create transferable models for impact assessment
        """)
    
    with col2:
        # Placeholder for an image or chart
        st.image("https://upload.wikimedia.org/wikipedia/commons/9/9a/TSMC_Fab_12.jpg", caption="TSMC Manufacturing Facility")
    
    st.markdown("---")
    
    st.subheader("Current Research Challenge")
    st.info("""
    While we've identified TSMC's Arizona facility as our target case study, 
    detailed data for this specific facility is currently unavailable as it's 
    still in development. However, comprehensive data for TSMC's Taiwan operations 
    is accessible through annual reports and public disclosures.
    """)

# Data Availability page
elif page == "Data Availability":
    st.header("Data Availability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Available TSMC Taiwan Data")
        
        # Create sample data for demonstration
        taiwan_data = {
            "Data Category": ["Production Volume", "Water Usage", "Energy Consumption", 
                             "Environmental Compliance", "Resource Efficiency", "Economic Impact"],
            "Availability": ["Complete", "Complete", "Complete", "Complete", "Complete", "Complete"],
            "Source": ["Annual Report", "ESG Report", "ESG Report", "Compliance Docs", "ESG Report", "Economic Analysis"],
            "Years": ["2018-2023", "2018-2023", "2018-2023", "2018-2023", "2018-2023", "2018-2023"]
        }
        
        df_taiwan = pd.DataFrame(taiwan_data)
        st.table(df_taiwan)
    
    with col2:
        st.subheader("Available TSMC Arizona Data")
        
        # Create sample data for demonstration
        arizona_data = {
            "Data Category": ["Production Volume", "Water Usage", "Energy Consumption", 
                             "Environmental Compliance", "Resource Efficiency", "Economic Impact"],
            "Availability": ["Limited", "Limited", "Limited", "Projected", "Limited", "Projected"],
            "Source": ["Press Releases", "Environmental Filings", "Projections", "Permit Applications", "Projections", "Economic Forecasts"],
            "Years": ["2023-2024", "2023-2024", "2023-2024", "2023-2024", "2023-2024", "2023-2024"]
        }
        
        df_arizona = pd.DataFrame(arizona_data)
        st.table(df_arizona)
    
    st.markdown("---")
    
    # Sample visualization
    st.subheader("Data Completeness Comparison")
    
    # Create data for visualization
    categories = ["Production", "Water", "Energy", "Compliance", "Resource", "Economic"]
    taiwan_completeness = [95, 90, 92, 88, 85, 80]
    arizona_completeness = [25, 30, 20, 35, 15, 40]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.35
    
    bar1 = ax.bar(x - width/2, taiwan_completeness, width, label='Taiwan Data')
    bar2 = ax.bar(x + width/2, arizona_completeness, width, label='Arizona Data')
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Data Completeness (%)')
    ax.set_title('Data Completeness by Category')
    ax.legend()
    
    st.pyplot(fig)
    
    st.markdown("""
    *Note: This visualization illustrates the current data availability gap between 
    our target case study (Arizona) and the available data source (Taiwan operations).*
    """)

# Methodology Approach page
elif page == "Methodology Approach":
    st.header("Proposed Methodology")
    
    st.image("https://miro.medium.com/max/1400/1*wUxzn_GPrgRR8OGJmTYO_w.png", caption="Model Development Process")
    
    st.markdown("""
    ### Two-Phase Approach
    
    Our proposed methodology involves a two-phase approach to overcome the data availability challenge:
    
    #### Phase 1: Model Development with Taiwan Data
    - Collect and clean comprehensive TSMC Taiwan operational data
    - Develop environmental-economic impact models
    - Validate models against historical Taiwan data
    - Assess model accuracy and reliability
    - Document transferability requirements
    
    #### Phase 2: Model Application to Arizona
    - Adapt validated models to Arizona context
    - Apply models with available Arizona data
    - Identify data gaps for further collection
    - Engage with TSMC management using validated model results
    - Refine models with new Arizona-specific data as it becomes available
    """)
    
    st.markdown("---")
    
    st.subheader("Model Components")
    
    components = {
        "Environmental Impact": ["Water usage intensity", "Energy consumption", "Carbon emissions", "Waste generation", "Chemical usage"],
        "Economic Impact": ["Direct employment", "Indirect employment", "Regional GDP contribution", "Tax revenue", "Supply chain effects"],
        "Resilience Metrics": ["Supply disruption risk", "Resource dependency", "Environmental compliance risk", "Technology adoption rate", "Regional economic integration"]
    }
    
    tab1, tab2, tab3 = st.tabs(list(components.keys()))
    
    with tab1:
        for item in components["Environmental Impact"]:
            st.markdown(f"- {item}")
    
    with tab2:
        for item in components["Economic Impact"]:
            st.markdown(f"- {item}")
    
    with tab3:
        for item in components["Resilience Metrics"]:
            st.markdown(f"- {item}")

# Model Validation Plan page
elif page == "Model Validation Plan":
    st.header("Model Validation Strategy")
    
    st.markdown("""
    ### Validation Approach
    
    To ensure our model is robust enough to transfer from Taiwan to Arizona operations,
    we will implement the following validation strategy:
    """)
    
    validation_steps = [
        "Historical data validation against Taiwan operations",
        "Sensitivity analysis for key environmental parameters",
        "Monte Carlo simulations for uncertainty quantification",
        "Cross-validation with industry benchmarks",
        "Expert panel review of model assumptions",
        "Geospatial factor adjustment for Arizona context"
    ]
    
    for i, step in enumerate(validation_steps):
        st.markdown(f"**Step {i+1}:** {step}")
    
    st.markdown("---")
    
    # Sample validation visualization
    st.subheader("Model Accuracy Simulation")
    
    # Create sample data for demonstration
    np.random.seed(42)
    actual = np.sort(np.random.normal(100, 15, 50))
    predicted = actual + np.random.normal(0, 10, 50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(actual, predicted, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Model Prediction vs Actual (Simulated)')
    
    st.pyplot(fig)
    
    # Show transferability factors
    st.subheader("Key Transferability Factors")
    
    transferability = {
        "Factor": ["Climate Conditions", "Regulatory Environment", "Water Availability", 
                  "Energy Mix", "Labor Market", "Supply Chain Structure"],
        "Taiwan Context": ["Humid subtropical", "Taiwan EPA regulations", "Abundant but regulated", 
                          "Nuclear and coal dominant", "Dense skilled workforce", "Asia-centric"],
        "Arizona Context": ["Hot desert", "US EPA and Arizona DEQ", "Water scarce region", 
                           "Mix of renewables and natural gas", "Developing semiconductor workforce", "North America-centric"],
        "Adjustment Need": ["High", "Medium", "High", "Medium", "Medium", "Low"]
    }
    
    df_transfer = pd.DataFrame(transferability)
    st.table(df_transfer)

# Expected Outcomes page
elif page == "Expected Outcomes":
    st.header("Expected Research Outcomes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Academic Outcomes")
        academic = [
            "Validated model for semiconductor environmental-economic assessment",
            "Methodology for cross-regional model transferability",
            "Quantification of semiconductor manufacturing environmental impacts",
            "Framework for supply chain resilience metrics",
            "Publication in sustainability and industrial ecology journals"
        ]
        
        for item in academic:
            st.markdown(f"- {item}")
    
    with col2:
        st.subheader("Practical Outcomes")
        practical = [
            "Decision support tool for TSMC Arizona operations",
            "Environmental impact forecasting capabilities",
            "Regional economic benefit assessment",
            "Supply chain vulnerability identification",
            "Policy recommendations for sustainable semiconductor manufacturing"
        ]
        
        for item in practical:
            st.markdown(f"- {item}")
    
    st.markdown("---")
    
    # Timeline
    st.subheader("Project Timeline")
    
    timeline = {
        "Phase": ["Taiwan Data Collection", "Model Development", "Taiwan Model Validation", 
                 "Arizona Model Adaptation", "Engagement with TSMC", "Arizona Data Collection",
                 "Final Model Validation", "Results Publication"],
        "Start": ["January 2025", "March 2025", "June 2025", "August 2025", 
                 "September 2025", "October 2025", "December 2025", "February 2026"],
        "End": ["February 2025", "May 2025", "July 2025", "September 2025", 
               "October 2025", "November 2025", "January 2026", "March 2026"],
        "Status": ["In Progress", "Planning", "Planning", "Planning", 
                  "Planning", "Planning", "Planning", "Planning"]
    }
    
    df_timeline = pd.DataFrame(timeline)
    
    fig = px.timeline(
        df_timeline, 
        x_start="Start", 
        x_end="End", 
        y="Phase",
        color="Status",
        title="Project Timeline"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Conclusion
    st.subheader("Key Proposal")
    st.success("""
    By developing and validating our model with Taiwan data first, we can:
    
    1. Establish methodological rigor and scientific validity
    2. Demonstrate value to TSMC management to facilitate Arizona data access
    3. Create a transferable framework for semiconductor sustainability assessment
    4. Bridge the current data gap while maintaining research momentum
    
    This approach ensures we can make progress despite limited Arizona-specific data
    while building toward a comprehensive assessment of the environmental-economic
    impacts of TSMC's Arizona operations.
    """)

# Add a contact form at the bottom
st.sidebar.markdown("---")
st.sidebar.subheader("Contact Information")
st.sidebar.markdown("Researcher Name: [Your Name]")
st.sidebar.markdown("Email: [your.email@university.edu]")
st.sidebar.markdown("Department: [Your Department]")

# Run this with: streamlit run tsmc_research_presentation.py
