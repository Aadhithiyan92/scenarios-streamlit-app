import streamlit as st

def inject_custom_css():
    """
    Injects custom CSS into the Streamlit app to simulate Tailwind-like styles.
    """
    st.markdown(
        """
        <style>
        /* Basic reset for body background */
        body {
            background-color: #F3F4F6; /* Tailwind gray-100 */
            margin: 0; 
            padding: 0;
        }

        /* Top bar / header */
        .my-top-bar {
            background: linear-gradient(to right, #2563EB, #1E40AF); /* Tailwind blues */
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .my-top-bar h1 {
            margin-bottom: 0;
            font-size: 2rem;
        }
        .my-top-bar p {
            margin: 0;
            font-size: 1rem;
            color: #93C5FD; /* lighter text */
        }

        /* Card-like container */
        .my-card {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        /* Sub-headers in content */
        .section-title {
            font-size: 1.5rem;
            color: #1E3A8A; /* Tailwind blue-800 */
            margin-bottom: 0.5rem;
            font-weight: bold;
        }

        /* Example color classes from Tailwind, simplified */
        .bg-blue-50 {
            background-color: #EFF6FF; 
            padding: 20px;
            border-radius: 10px;
        }
        .bg-green-50 {
            background-color: #ECFDF5;
            padding: 20px;
            border-radius: 10px;
        }
        .text-blue-800 {
            color: #1E3A8A;
        }
        .text-green-800 {
            color: #065F46;
        }

        /* Slightly styled list */
        .custom-list li {
            margin-bottom: 0.5rem;
        }

        /* A 'card row' container for multi-column layout if you want custom styling */
        .card-row {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .card {
            flex: 1;
            min-width: 250px;
            background-color: #FFFFFF;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Configure basic page settings
    st.set_page_config(page_title="Environmental-Economic Sensitivity Analysis", layout="wide")
    
    # Inject the custom CSS
    inject_custom_css()

    # Top Bar / Header
    st.markdown(
        """
        <div class="my-top-bar">
            <h1>Environmental-Economic Sensitivity Analysis</h1>
            <p>US Regional Semiconductor Supply Chain Dynamics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar menu
    menu = ["Overview", "Methodology", "Regions", "Data", "Equations", "Results"]
    choice = st.sidebar.radio("Sections", menu)

    # Render chosen section
    if choice == "Overview":
        overview_section()
    elif choice == "Methodology":
        methodology_section()
    elif choice == "Regions":
        regions_section()
    elif choice == "Data":
        data_section()
    elif choice == "Equations":
        equations_section()
    elif choice == "Results":
        results_section()

def overview_section():
    st.markdown('<div class="my-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Research Overview</div>', unsafe_allow_html=True)
    st.write("""
    This cutting-edge research investigates the complex dynamics between environmental 
    sustainability and semiconductor supply chain resilience across key US manufacturing regions. 
    Using advanced nonlinear dynamical systems analysis, we model the intricate relationships 
    between water availability, energy transitions, and environmental regulations.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Using columns for a two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="my-card bg-blue-50">', unsafe_allow_html=True)
        st.markdown('<div class="section-title text-blue-800">Key Innovations</div>', unsafe_allow_html=True)
        st.markdown('<ul class="custom-list"><li>Nonlinear coupling of environmental-economic factors</li><li>Regional sensitivity analysis framework</li><li>Critical threshold identification methods</li></ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="my-card bg-green-50">', unsafe_allow_html=True)
        st.markdown('<div class="section-title text-green-800">Expected Impact</div>', unsafe_allow_html=True)
        st.markdown('<ul class="custom-list"><li>Enhanced supply chain resilience strategies</li><li>Regional policy recommendations</li><li>Sustainability-oriented manufacturing practices</li></ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def regions_section():
    st.markdown('<div class="section-title">Regional Analysis</div>', unsafe_allow_html=True)
    st.write("Key manufacturing regions and their critical environmental-economic characteristics:")

    st.markdown('<div class="card-row">', unsafe_allow_html=True)

    st.markdown('''
        <div class="card">
            <h3 class="text-blue-800">Southwest Region (AZ, NM)</h3>
            <ul>
                <li>Water scarcity challenges</li>
                <li>High solar energy potential</li>
                <li>Major players: Intel, TSMC</li>
            </ul>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="card">
            <h3 class="text-blue-800">Pacific Northwest (OR, WA)</h3>
            <ul>
                <li>Hydroelectric power availability</li>
                <li>Stable water supply</li>
                <li>Major player: Intel</li>
            </ul>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="card">
            <h3 class="text-blue-800">Texas Region</h3>
            <ul>
                <li>Independent power grid (ERCOT)</li>
                <li>Mixed energy sources</li>
                <li>Major players: Samsung, TI</li>
            </ul>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="card">
            <h3 class="text-blue-800">Northeast Corridor (NY)</h3>
            <ul>
                <li>Stable water resources</li>
                <li>Strict environmental regulations</li>
                <li>Major player: GlobalFoundries</li>
            </ul>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def data_section():
    st.markdown('<div class="section-title">Data Requirements & Accessibility</div>', unsafe_allow_html=True)
    st.write("Below is an overview of easily accessible, moderately difficult, and challenging data sources.")

    # 3 column layout for data categories
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="my-card bg-green-50">', unsafe_allow_html=True)
        st.markdown('<div class="section-title text-green-800" style="font-size:1.2rem;">Easily Accessible Data</div>', unsafe_allow_html=True)
        st.markdown('<ul class="custom-list"><li>Regional Energy Consumption (DOE)</li><li>Water Usage Permits (State Environmental Agencies)</li><li>Environmental Compliance Records (EPA Database)</li></ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="my-card" style="background-color: #FEFCE8;">', unsafe_allow_html=True)  # Approx. yellow-50
        st.markdown('<div class="section-title" style="color: #854D0E; font-size:1.2rem;">Moderately Difficult Data</div>', unsafe_allow_html=True)
        st.markdown('<ul class="custom-list"><li>Facility Energy Usage (Company Reports)</li><li>Water Recycling Rates (Industry Surveys)</li><li>Production Capacity (Industry Reports)</li></ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="my-card" style="background-color: #FEE2E2;">', unsafe_allow_html=True)  # Approx. red-100
        st.markdown('<div class="section-title" style="color: #7F1D1D; font-size:1.2rem;">Challenging Data</div>', unsafe_allow_html=True)
        st.markdown('<ul class="custom-list"><li>Efficiency Metrics (Proprietary Data)</li><li>Production Costs (Internal Records)</li><li>Environmental Targets (Corporate Plans)</li></ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def methodology_section():
    st.markdown('<div class="section-title">Research Methodology</div>', unsafe_allow_html=True)
    st.write("""
    We employ advanced system dynamics and nonlinear analysis techniques to model interdependencies 
    among water availability, energy consumption, and production throughput in semiconductor 
    manufacturing across multiple US regions.
    """)

    # System Relationships
    st.markdown("#### System Relationships")
    st.markdown("*(Using a placeholder diagram URL. Replace with your own.)*")
    st.image("https://via.placeholder.com/600x300.png?text=System+Relationships", use_column_width=True)

    # Analysis Methods & Implementation Steps
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="my-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analysis Methods</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Lyapunov Stability Analysis**: Examining system stability near equilibrium points  
            - **Bifurcation Analysis**: Identifying critical parameter thresholds  
            - **Sensitivity Analysis**: Parameter impact assessment  
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="my-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Implementation Steps</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Data Collection & Validation**: Regional environmental and production data  
            - **Model Calibration**: Parameter estimation and validation  
            - **Regional Analysis**: Comparative regional studies  
            """)
        st.markdown('</div>', unsafe_allow_html=True)

def equations_section():
    st.markdown('<div class="section-title">System Equations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Below is a conceptual overview of a nonlinear dynamical system describing production (P), 
        water (W), energy (E), etc. We'll use ASCII-friendly variables to avoid syntax issues:
        """
    )

    st.code(
        """
# Example (ASCII-friendly):
# dP/dt = mu1 * M(t)*E(t)*W(t)*(1 - P/K) - delta1 * D(t)*P^2
# dW/dt = alpha2 * P(t)*(1 - W/Wmax) - beta2 * R(t)*W^2 - delta2 * T(t)
# dE/dt = [alpha1 * P(t) + beta1 * M(t)](1 - E/Emax) - gamma1 * R(t)*E^2
# ...
        """,
        language="python"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="my-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Stability Analysis</div>', unsafe_allow_html=True)
        st.markdown("- Lyapunov stability analysis\n- Bifurcation analysis\n- Phase space analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="my-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Parameters</div>', unsafe_allow_html=True)
        st.markdown("- Environmental coupling coefficients\n- Production efficiency factors\n- Resource utilization rates")
        st.markdown('</div>', unsafe_allow_html=True)

def results_section():
    st.markdown('<div class="section-title">Expected Results & Impact</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="my-card bg-blue-50">', unsafe_allow_html=True)
        st.markdown('<div class="section-title text-blue-800">Research Outcomes</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Regional Stability Maps**: Identify critical thresholds for each region  
            - **Sensitivity Metrics**: Measure system response to parameter variations  
            - **Risk Assessment Framework**: Evaluate environmental risks to supply chain stability  
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="my-card bg-green-50">', unsafe_allow_html=True)
        st.markdown('<div class="section-title text-green-800">Expected Impact</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Policy Recommendations**: Guidance for regional policymakers  
            - **Industry Guidelines**: Best practices for environmental sustainability  
            - **Resilience Strategies**: Actionable plans for supply chain strengthening  
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
