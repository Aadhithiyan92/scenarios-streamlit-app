import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(layout="wide", page_title="Semiconductor Supply Chain Analysis")

# Title and Introduction
st.title("Environmental-Economic Sensitivity Analysis of US Regional Semiconductor Supply Chains")

# Sidebar for Navigation
page = st.sidebar.selectbox(
    "Select Section",
    ["Overview", "Objectives", "Methodology", "Data Requirements", "Regional Analysis", "Results"]
)

if page == "Overview":
    st.header("Research Overview")
    
    # Background and Motivation
    st.subheader("Background & Motivation")
    st.write("""
    The US semiconductor industry faces unprecedented challenges in maintaining supply chain resilience 
    while meeting environmental sustainability goals. Recent events have highlighted critical vulnerabilities:
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        🔸 **Supply Chain Crisis**
        - 2020-2023 global chip shortage
        - Production bottlenecks
        - Regional concentration risks
        - Economic impacts
        
        🔸 **Environmental Pressures**
        - Increasing water scarcity
        - Rising energy demands
        - Climate change impacts
        - Resource constraints
        """)
    with col2:
        st.markdown("""
        🔸 **Regulatory Changes**
        - Stricter environmental policies
        - Sustainability requirements
        - Carbon reduction targets
        - Water usage regulations
        
        🔸 **Regional Dependencies**
        - Water-stressed manufacturing zones
        - Energy grid vulnerabilities
        - Resource availability variations
        - Geographic concentration risks
        """)

    # Previous Approaches
    st.subheader("Previous Approaches & Limitations")
    with st.expander("See previous approaches and their limitations"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Traditional Approaches:**
            - Linear supply chain models
            - Static analysis methods
            - Isolated regional studies
            - Single-factor analysis
            """)
        with col2:
            st.markdown("""
            **Key Limitations:**
            - Failed to capture complex interactions
            - Missed dynamic system evolution
            - Overlooked environmental coupling
            - Ignored regional interdependencies
            """)

    # Our Novel Approach
    st.subheader("Our Novel Approach")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Key Innovations")
        st.markdown("""
        - Nonlinear coupling of environmental-economic factors
        - Regional sensitivity analysis framework
        - Critical threshold identification methods
        - Advanced stability analysis techniques
        - Dynamic system evolution modeling
        - Multi-regional interaction analysis
        """)
    with col2:
        st.markdown("#### Expected Impact")
        st.markdown("""
        - Enhanced supply chain resilience strategies
        - Regional policy recommendations
        - Sustainability-oriented manufacturing practices
        - Improved risk assessment frameworks
        - Evidence-based decision support
        - Adaptive management capabilities
        """)

    # Research Significance
    st.subheader("Research Significance & Timing")
    with st.expander("Why this research is crucial now"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Current Drivers:**
            - CHIPS Act Implementation
            - Climate Change Impacts
            - Supply Chain Vulnerabilities
            - Environmental Regulations
            """)
        with col2:
            st.markdown("""
            **Strategic Value:**
            - Guide manufacturing location decisions
            - Enhance resource management
            - Improve resilience planning
            - Support policy development
            """)

    # Research Impact
    st.subheader("Expected Research Impact")
    impact_cols = st.columns(3)
    with impact_cols[0]:
        st.markdown("**Industry Impact**")
        st.markdown("""
        - Better decision-making tools
        - Resource optimization
        - Risk mitigation strategies
        """)
    with impact_cols[1]:
        st.markdown("**Policy Guidance**")
        st.markdown("""
        - Evidence-based recommendations
        - Balanced growth frameworks
        - Sustainability guidelines
        """)
    with impact_cols[2]:
        st.markdown("**Academic Contribution**")
        st.markdown("""
        - Novel mathematical framework
        - Analytical methodology
        - Future research directions
        """)
 # Research Architecture
    st.subheader("Research Architecture")
    tab1, tab2, tab3 = st.tabs(["Why This Research?", "How We Approach It", "Research Flow"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### Critical Industry Challenges
            
            🏭 **Manufacturing Sector Impact**
            - $500B+ global semiconductor market
            - Critical for technology advancement
            - Essential for economic security
            
            🌍 **Environmental Concerns**
            - High water consumption (10M+ gallons/day)
            - Significant energy usage
            - Environmental regulation compliance
            
            🔄 **Supply Chain Vulnerabilities**
            - Geographic concentration risks
            - Resource dependency issues
            - Production bottlenecks
            """)
        
        with col2:
            st.markdown("""
            ### Why Dynamic Systems Approach?
            
            📊 **Complex Interactions**
            - Nonlinear relationships
            - Multiple feedback loops
            - Time-dependent behavior
            
            🎯 **Predictive Capabilities**
            - Early warning indicators
            - Stability assessment
            - Risk prediction
            
            🔍 **System Understanding**
            - Root cause analysis
            - Parameter sensitivity
            - Critical thresholds
            """)
    
    with tab2:
        st.markdown("""
        ### Our Methodological Framework
        
        #### 1. System Modeling
        - Nonlinear differential equations
        - Coupled environmental-economic variables
        - Regional parameter variations
        
        #### 2. Analysis Techniques
        - Stability analysis using Lyapunov methods
        - Bifurcation analysis for critical points
        - Sensitivity analysis for key parameters
        
        #### 3. Validation & Implementation
        - Data-driven validation
        - Regional case studies
        - Industry feedback integration
        """)
        
        st.markdown("### Key Mathematical Components")
        math_cols = st.columns(3)
        with math_cols[0]:
            st.markdown("""
            **State Variables**
            - Production Capacity (P)
            - Water Availability (W)
            - Energy Supply (E)
            - Resource Efficiency (R)
            """)
        with math_cols[1]:
            st.markdown("""
            **Control Parameters**
            - Production rates
            - Resource utilization
            - Environmental limits
            - Efficiency factors
            """)
        with math_cols[2]:
            st.markdown("""
            **Output Metrics**
            - Stability indices
            - Risk indicators
            - Performance measures
            - Resilience metrics
            """)
    
    with tab3:
        st.markdown("### Research Architecture and Flow")
        
        # Create a visual flow using columns and styled boxes
        st.markdown("""
        <style>
        .research-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        .phase-1 { background-color: #f9d5e5; }
        .phase-2 { background-color: #eff6ff; }
        .phase-3 { background-color: #dcfce7; }
        .phase-4 { background-color: #fef3c7; }
        </style>
        """, unsafe_allow_html=True)

        # Phase 1: Problem Identification
        st.markdown("""
        <div class="research-box phase-1">
            <h4>Phase 1: Problem Identification</h4>
            <p>• Supply Chain Vulnerabilities<br>
               • Environmental Challenges<br>
               • Regional Dependencies<br>
               • Regulatory Pressures</p>
        </div>
        """, unsafe_allow_html=True)

        # Phase 2: Methodology Development
        st.markdown("""
        <div class="research-box phase-2">
            <h4>Phase 2: Methodology Development</h4>
            <p>• Nonlinear Dynamic Modeling<br>
               • Regional Sensitivity Analysis<br>
               • Environmental-Economic Coupling</p>
        </div>
        """, unsafe_allow_html=True)

        # Phase 3: Implementation
        st.markdown("""
        <div class="research-box phase-3">
            <h4>Phase 3: Implementation</h4>
            <p>• Mathematical Framework<br>
               • Data Analysis<br>
               • Stability Assessment</p>
        </div>
        """, unsafe_allow_html=True)

        # Phase 4: Impact
        st.markdown("""
        <div class="research-box phase-4">
            <h4>Phase 4: Expected Impact</h4>
            <p>• Industry Guidelines<br>
               • Policy Recommendations<br>
               • Resilience Strategies</p>
        </div>
        """, unsafe_allow_html=True)

        # Implementation Steps
        st.markdown("### Detailed Implementation Process")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Technical Implementation
            1. **System Characterization**
               - Variable identification
               - Relationship mapping
               - Boundary definition
            
            2. **Model Development**
               - Equation formulation
               - Parameter definition
               - Constraint establishment
            """)
            
        with col2:
            st.markdown("""
            #### Analysis & Output
            3. **Analysis & Validation**
               - Stability analysis
               - Sensitivity testing
               - Regional validation
            
            4. **Results & Recommendations**
               - Policy guidelines
               - Industry recommendations
               - Implementation strategies
            """)

        # Key Outcomes
        st.markdown("### Key Research Outcomes")
        outcomes_cols = st.columns(3)
        
        with outcomes_cols[0]:
            st.markdown("""
            🎯 **Technical Outcomes**
            - Mathematical models
            - Analysis frameworks
            - Validation methods
            """)
            
        with outcomes_cols[1]:
            st.markdown("""
            📊 **Practical Outputs**
            - Decision support tools
            - Risk assessment methods
            - Performance metrics
            """)
            
        with outcomes_cols[2]:
            st.markdown("""
            🌍 **Impact Areas**
            - Industry practices
            - Policy development
            - Sustainability goals
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
