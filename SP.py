import streamlit as st

# 1) Inject custom CSS to style the page.
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Global body style */
        body {
            margin: 0;
            padding: 0;
            background-color: #F9FAFB; /* Light gray background */
        }

        /* Remove default Streamlit padding */
        .main .block-container {
            padding: 0 !important;
        }

        /* Top bar container */
        .top-bar {
            background: linear-gradient(to right, #3B82F6, #2563EB); /* Tailwind blues */
            color: white;
            padding: 1.5rem;
            border-radius: 0 0 8px 8px;
            margin-bottom: 1rem;
        }
        .top-bar h1 {
            margin: 0;
            font-size: 1.8rem;
        }
        .top-bar p {
            margin: 0;
            color: #DBEAFE; /* lighter text */
        }

        /* Layout container for the sidebar and main content */
        .layout-container {
            display: flex;
        }

        /* Custom sidebar styling */
        .sidebar-container {
            width: 220px;
            background-color: #FFFFFF;
            border-right: 1px solid #E5E7EB;
            min-height: calc(100vh - 80px); /* minus top bar */
            padding-top: 1rem;
        }

        /* Remove default Streamlit sidebar */
        section[data-testid="stSidebar"] {
            display: none;
        }

        /* Sidebar nav items container */
        .nav-list {
            list-style-type: none;
            padding: 0;
        }
        .nav-item {
            padding: 0.75rem 1rem;
            margin: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            color: #4B5563; /* Gray-600 */
            transition: background-color 0.2s;
        }
        .nav-item:hover {
            background-color: #F3F4F6; /* Gray-100 */
        }
        .nav-item.active {
            background-color: #2563EB; /* Blue-600 */
            color: white !important;
        }

        /* Main content area */
        .main-content {
            flex: 1;
            padding: 1rem 2rem; 
        }

        /* Section titles (blue) */
        .section-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1E3A8A; /* Blue-800 */
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        /* Card-like containers */
        .card {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .card h3 {
            margin-top: 0;
        }
        .card-list {
            list-style-type: none;
            padding-left: 1.5rem;
        }
        .card-list li {
            margin-bottom: 0.5rem;
        }

        /* Some color utility classes (approx. Tailwind) */
        .bg-blue-50 { background-color: #EFF6FF; }
        .bg-green-50 { background-color: #ECFDF5; }
        .bg-yellow-50 { background-color: #FFFBEB; }
        .bg-red-50 { background-color: #FEF2F2; }

        .text-blue-800 { color: #1E3A8A; }
        .text-green-800 { color: #065F46; }
        .text-yellow-800 { color: #854D0E; }
        .text-red-800 { color: #7F1D1D; }

        /* Headings inside cards */
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# 2) Our main app function
def main():
    st.set_page_config(page_title="Environmental-Economic Sensitivity Analysis", layout="wide")
    inject_custom_css()

    # We'll track the selected page in st.session_state
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Overview"

    # 2A) Top bar
    st.markdown(
        """
        <div class="top-bar">
            <h1>Environmental-Economic Sensitivity Analysis</h1>
            <p>US Regional Semiconductor Supply Chain Dynamics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2B) Body layout: custom 'layout-container' with a custom "sidebar" div plus main content
    st.markdown('<div class="layout-container">', unsafe_allow_html=True)

    # Sidebar
    st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
    render_sidebar()
    st.markdown('</div>', unsafe_allow_html=True)

    # Main Content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    render_page(st.session_state.selected_page)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# 3) Render the custom sidebar
def render_sidebar():
    pages = ["Overview", "Methodology", "Regions", "Data", "Equations", "Results"]

    st.markdown('<ul class="nav-list">', unsafe_allow_html=True)
    for page in pages:
        # Check if this page is currently active
        active_class = "active" if page == st.session_state.selected_page else ""
        sidebar_item = f"""
        <li class="nav-item {active_class}" onClick="window.location.href='?selected_page={page}'">
            {page}
        </li>
        """
        st.markdown(sidebar_item, unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)

# 4) Based on selected_page, display the appropriate section
def render_page(page):
    # When a nav item is clicked, we get a query param '?selected_page=PageName'
    # Let's read that param to update session_state
    query_params = st.experimental_get_query_params()
    if "selected_page" in query_params:
        st.session_state.selected_page = query_params["selected_page"][0]

    if page == "Overview":
        overview_section()
    elif page == "Methodology":
        methodology_section()
    elif page == "Regions":
        regions_section()
    elif page == "Data":
        data_section()
    elif page == "Equations":
        equations_section()
    elif page == "Results":
        results_section()

def overview_section():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Research Overview</div>', unsafe_allow_html=True)
    st.write(
        """
        This cutting-edge research investigates the complex dynamics between environmental 
        sustainability and semiconductor supply chain resilience across key US manufacturing regions. 
        Using advanced nonlinear dynamical systems analysis, we model the intricate relationships 
        between water availability, energy transitions, and environmental regulations.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # "Key Innovations" (blue) and "Expected Impact" (green) side by side using st.columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card bg-blue-50">', unsafe_allow_html=True)
        st.markdown('<div class="card-title text-blue-800">Key Innovations</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="card-list">
            <li>Nonlinear coupling of environmental-economic factors</li>
            <li>Regional sensitivity analysis framework</li>
            <li>Critical threshold identification methods</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card bg-green-50">', unsafe_allow_html=True)
        st.markdown('<div class="card-title text-green-800">Expected Impact</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="card-list">
            <li>Enhanced supply chain resilience strategies</li>
            <li>Regional policy recommendations</li>
            <li>Sustainability-oriented manufacturing practices</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

def methodology_section():
    st.markdown('<div class="section-title">Research Methodology</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <p>We employ advanced system dynamics and nonlinear analysis techniques to model 
            interdependencies among water availability, energy consumption, and production throughput 
            in semiconductor manufacturing across multiple US regions.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # System Relationships + columns for Analysis Methods & Implementation Steps
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">System Relationships</div>', unsafe_allow_html=True)
    st.markdown("*(Placeholder for your diagram—replace with an actual image URL or remove.)*")
    st.image("https://via.placeholder.com/600x300.png?text=System+Relationships", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Analysis Methods</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="card-list">
            <li><strong>Lyapunov Stability Analysis</strong>: Examine system stability near equilibrium points</li>
            <li><strong>Bifurcation Analysis</strong>: Identify critical parameter thresholds</li>
            <li><strong>Sensitivity Analysis</strong>: Assess parameter impact</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Implementation Steps</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="card-list">
            <li><strong>Data Collection & Validation</strong>: Environmental & production data</li>
            <li><strong>Model Calibration</strong>: Parameter estimation & validation</li>
            <li><strong>Regional Analysis</strong>: Comparative regional studies</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

def regions_section():
    st.markdown('<div class="section-title">Regional Analysis</div>', unsafe_allow_html=True)

    # Each region as a card
    st.markdown(
        """
        <div class="card">
            <h3 class="text-blue-800">Southwest Region (AZ, NM)</h3>
            <ul class="card-list">
                <li>Water scarcity challenges</li>
                <li>High solar energy potential</li>
                <li>Major players: Intel, TSMC</li>
            </ul>
        </div>

        <div class="card">
            <h3 class="text-blue-800">Pacific Northwest (OR, WA)</h3>
            <ul class="card-list">
                <li>Hydroelectric power availability</li>
                <li>Stable water supply</li>
                <li>Major player: Intel</li>
            </ul>
        </div>

        <div class="card">
            <h3 class="text-blue-800">Texas Region</h3>
            <ul class="card-list">
                <li>Independent power grid (ERCOT)</li>
                <li>Mixed energy sources</li>
                <li>Major players: Samsung, TI</li>
            </ul>
        </div>

        <div class="card">
            <h3 class="text-blue-800">Northeast Corridor (NY)</h3>
            <ul class="card-list">
                <li>Stable water resources</li>
                <li>Strict environmental regulations</li>
                <li>Major player: GlobalFoundries</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

def data_section():
    st.markdown('<div class="section-title">Data Requirements & Accessibility</div>', unsafe_allow_html=True)

    # Three data categories as separate cards
    st.markdown('<div class="card bg-green-50">', unsafe_allow_html=True)
    st.markdown('<div class="card-title text-green-800">Easily Accessible Data</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="card-list">
            <li><strong>Regional Energy Consumption</strong><br/><em>Source: Department of Energy (DOE)</em></li>
            <li><strong>Water Usage Permits</strong><br/><em>Source: State Environmental Agencies</em></li>
            <li><strong>Environmental Compliance Records</strong><br/><em>Source: EPA Database</em></li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card bg-yellow-50">', unsafe_allow_html=True)
    st.markdown('<div class="card-title text-yellow-800">Moderately Difficult Data</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="card-list">
            <li><strong>Facility Energy Usage</strong><br/><em>Source: Company Reports</em></li>
            <li><strong>Water Recycling Rates</strong><br/><em>Source: Industry Surveys</em></li>
            <li><strong>Production Capacity</strong><br/><em>Source: Industry Reports</em></li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card bg-red-50">', unsafe_allow_html=True)
    st.markdown('<div class="card-title text-red-800">Challenging Data</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="card-list">
            <li><strong>Efficiency Metrics</strong><br/><em>Source: Proprietary Data</em></li>
            <li><strong>Production Costs</strong><br/><em>Source: Internal Records</em></li>
            <li><strong>Environmental Targets</strong><br/><em>Source: Corporate Plans</em></li>
        </ul>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

def equations_section():
    st.markdown('<div class="section-title">System Equations</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        <p>Below is a conceptual overview of a nonlinear dynamical system describing production (P),
        water (W), energy (E), etc. in ASCII-friendly notation:</p>
        """,
        unsafe_allow_html=True
    )
    st.code(
        """
dP/dt = mu1 * M(t)*E(t)*W(t)*(1 - P/K) - delta1 * D(t)*P^2
dW/dt = alpha2 * P(t)*(1 - W/Wmax) - beta2 * R(t)*W^2 - delta2 * T(t)
dE/dt = [alpha1*P(t) + beta1*M(t)](1 - E/Emax) - gamma1*R(t)*E^2
...
        """,
        language="python"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Stability Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="card-list">
                <li>Lyapunov stability analysis</li>
                <li>Bifurcation analysis</li>
                <li>Phase space analysis</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Parameters</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="card-list">
                <li>Environmental coupling coefficients</li>
                <li>Production efficiency factors</li>
                <li>Resource utilization rates</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

def results_section():
    st.markdown('<div class="section-title">Expected Results & Impact</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card bg-blue-50">', unsafe_allow_html=True)
        st.markdown('<div class="card-title text-blue-800">Research Outcomes</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="card-list">
                <li><strong>Regional Stability Maps</strong><br/>Identification of stability boundaries and critical thresholds</li>
                <li><strong>Sensitivity Metrics</strong><br/>System response to parameter variations</li>
                <li><strong>Risk Assessment Framework</strong><br/>Evaluation of environmental risks to supply chain stability</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card bg-green-50">', unsafe_allow_html=True)
        st.markdown('<div class="card-title text-green-800">Expected Impact</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <ul class="card-list">
                <li><strong>Policy Recommendations</strong><br/>Evidence-based guidance for regional policymakers</li>
                <li><strong>Industry Guidelines</strong><br/>Best practices for environmental sustainability</li>
                <li><strong>Resilience Strategies</strong><br/>Actionable plans for supply chain strengthening</li>
            </ul>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# 5) Run the app
if __name__ == "__main__":
    main()
