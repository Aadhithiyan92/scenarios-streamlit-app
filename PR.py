import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(layout="wide", page_title="Semiconductor Supply Chain Analysis")

# Title and Introduction
st.title("Environmental-Economic Sensitivity Analysis of US Regional Semiconductor Supply Chains")

# Sidebar Navigation
page = st.sidebar.selectbox(
    "Select Section",
    ["Overview", "Research Architecture", "Objectives", "Methodology", "Data Requirements", "Regional Analysis", "Results"]
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
# Environmental-Economic Relationship
    st.subheader("Environmental-Economic Relationship")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        🌍 **Environmental Factors**
        - Water Availability
          • Manufacturing needs ~10M gallons/day
          • Water scarcity issues
          • Quality requirements
        
        - Energy Resources
          • High power consumption
          • Grid reliability
          • Clean energy transition
        
        - Climate Impact
          • Temperature control needs
          • Extreme weather risks
          • Cooling requirements
        """)
    
    with col2:
        st.markdown("""
        💰 **Economic Impacts**
        - Cost Effects
          • Higher water procurement costs
          • Energy price fluctuations
          • Infrastructure investments
        
        - Production Impact
          • Capacity limitations
          • Efficiency losses
          • Operating cost increases
        
        - Investment Needs
          • Recycling systems
          • Backup power solutions
          • Technology upgrades
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

elif page == "Research Architecture":
    st.header("Research Architecture and Flow")
    
    # Create visual flow using styled boxes
    st.markdown("""
    <style>
    .research-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .phase-1 { background-color: #f9d5e5; }
    .phase-2 { background-color: #eff6ff; }
    .phase-3 { background-color: #dcfce7; }
    .phase-4 { background-color: #fef3c7; }
    </style>
    """, unsafe_allow_html=True)

    # Research Flow Visualization
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Research Flow")
        st.markdown("""
        This diagram shows the complete
        flow of our research from problem
        identification to impact generation.
        """)
    
    with col2:
        # Phase boxes
        st.markdown("""
        <div class="research-box phase-1">
            <h4>Phase 1: Problem Identification</h4>
            <p>• Supply Chain Vulnerabilities<br>
               • Environmental Challenges<br>
               • Regional Dependencies<br>
               • Regulatory Pressures</p>
        </div>
        
        <div class="research-box phase-2">
            <h4>Phase 2: Our Approach</h4>
            <p>• Nonlinear Dynamic Modeling<br>
               • Regional Sensitivity Analysis<br>
               • Environmental-Economic Coupling</p>
        </div>
        
        <div class="research-box phase-3">
            <h4>Phase 3: Implementation</h4>
            <p>• Mathematical Framework<br>
               • Data Analysis<br>
               • Stability Assessment</p>
        </div>
        
        <div class="research-box phase-4">
            <h4>Phase 4: Expected Impact</h4>
            <p>• Industry Guidelines<br>
               • Policy Recommendations<br>
               • Resilience Strategies</p>
        </div>
        """, unsafe_allow_html=True)

    # Methodology Details
    st.subheader("Detailed Methodology")
    method_col1, method_col2, method_col3 = st.columns(3)
    
    with method_col1:
        st.markdown("""
        #### System Modeling
        - Nonlinear differential equations
        - Environmental coupling
        - Regional parameters
        """)
        
    with method_col2:
        st.markdown("""
        #### Analysis Techniques
        - Stability analysis
        - Sensitivity testing
        - Bifurcation analysis
        """)
        
    with method_col3:
        st.markdown("""
        #### Validation Process
        - Data verification
        - Regional case studies
        - Industry feedback
        """)

    # Key Components
    st.subheader("Research Components")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("""
        ### Technical Framework
        1. **Mathematical Model**
           - State variables definition
           - System equations
           - Parameter identification
        
        2. **Analysis Methods**
           - Stability assessment
           - Sensitivity analysis
           - Regional comparison
        """)
        
    with comp_col2:
        st.markdown("""
        ### Implementation Strategy
        1. **Data Collection**
           - Environmental metrics
           - Production data
           - Regional characteristics
        
        2. **Validation Process**
           - Model verification
           - Results validation
           - Impact assessment
        """)
elif page == "Objectives":
    st.header("Research Objectives")
    
    # Main Research Goal
    st.markdown("""
    <div style='background-color: #f0f9ff; padding: 20px; border-radius: 10px; border-left: 5px solid #0369a1;'>
        <h3 style='color: #0369a1; margin-top: 0;'>Primary Research Goal</h3>
        <p>Develop a comprehensive framework for analyzing and enhancing the environmental-economic 
        resilience of US semiconductor supply chains through nonlinear dynamical systems modeling.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Specific Objectives with icons
    st.subheader("Specific Objectives")
    
    objectives_data = {
        "Modeling & Analysis": {
            "icon": "📊",
            "objectives": [
                "Develop nonlinear dynamical system models",
                "Implement regional sensitivity analysis",
                "Identify critical thresholds",
                "Analyze stability boundaries"
            ],
            "color": "#bae6fd"
        },
        "Environmental Integration": {
            "icon": "🌍",
            "objectives": [
                "Quantify environmental constraints",
                "Model resource dependencies",
                "Assess sustainability metrics",
                "Evaluate ecological impacts"
            ],
            "color": "#bbf7d0"
        },
        "Regional Assessment": {
            "icon": "🏢",
            "objectives": [
                "Compare regional characteristics",
                "Analyze geographic variations",
                "Evaluate resource availability",
                "Assess regional risks"
            ],
            "color": "#fde68a"
        },
        "Policy & Implementation": {
            "icon": "📋",
            "objectives": [
                "Develop policy recommendations",
                "Create implementation guidelines",
                "Design resilience strategies",
                "Establish monitoring frameworks"
            ],
            "color": "#fecaca"
        }
    }
    
    # Create a 2x2 grid of objective categories
    col1, col2 = st.columns(2)
    cols = [col1, col2] * 2  # Repeat columns for 4 items
    
    for idx, (category, data) in enumerate(objectives_data.items()):
        with cols[idx]:
            st.markdown(f"""
            <div style='background-color: {data["color"]}; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                <h3>{data["icon"]} {category}</h3>
                <ul style='list-style-type: none; padding-left: 0;'>
                    {"".join(f"<li style='margin: 10px 0;'>• {obj}</li>" for obj in data["objectives"])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Expected Outcomes
    st.subheader("Expected Outcomes")
    outcomes_col1, outcomes_col2, outcomes_col3 = st.columns(3)
    
    with outcomes_col1:
        st.markdown("""
        #### 🎯 Technical Outcomes
        - Mathematical models
        - Analysis frameworks
        - Validation methods
        """)
    
    with outcomes_col2:
        st.markdown("""
        #### 📈 Practical Outputs
        - Decision support tools
        - Risk assessment methods
        - Performance metrics
        """)
    
    with outcomes_col3:
        st.markdown("""
        #### 🌟 Impact Areas
        - Industry practices
        - Policy development
        - Sustainability goals
        """)

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
