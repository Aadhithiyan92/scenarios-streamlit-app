import streamlit as st

def inject_custom_css():
    """
    Inject custom CSS to style the page, including
    new classes for system formation cards.
    """
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

        /* Hide default Streamlit sidebar */
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

        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        /* New system-formation classes */
        .system-card {
            background-color: #FFFFFF;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .system-card-title {
            color: #1E3A8A;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .system-list {
            list-style-type: none;
            padding-left: 0;
        }
        .system-list li {
            margin-bottom: 1rem;
            padding-left: 1.5rem;
            position: relative;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Environmental-Economic Sensitivity Analysis", layout="wide")

    # Inject our custom CSS
    inject_custom_css()

    # We track selected page in session_state
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Overview"

    # Top Bar
    st.markdown(
        """
        <div class="top-bar">
            <h1>Environmental-Economic Sensitivity Analysis</h1>
            <p>US Regional Semiconductor Supply Chain Dynamics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Body layout: custom "layout-container" with a custom sidebar plus main content
    st.markdown('<div class="layout-container">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
    render_sidebar()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    render_page(st.session_state.selected_page)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar():
    # Updated pages list to include "System Formation"
    pages = ["Overview", "Methodology", "System Formation", "Regions", "Data", "Equations", "Results"]

    st.markdown('<ul class="nav-list">', unsafe_allow_html=True)
    for page in pages:
        active_class = "active" if page == st.session_state.selected_page else ""
        sidebar_item = f"""
        <li class="nav-item {active_class}" onClick="window.location.href='?selected_page={page}'">
            {page}
        </li>
        """
        st.markdown(sidebar_item, unsafe_allow_html=True)
    st.markdown('</ul>', unsafe_allow_html=True)

def render_page(page):
    query_params = st.experimental_get_query_params()
    if "selected_page" in query_params:
        st.session_state.selected_page = query_params["selected_page"][0]

    if page == "Overview":
        overview_section()
    elif page == "Methodology":
        methodology_section()
    elif page == "System Formation":
        system_formation()
    elif page == "Regions":
        regions_section()
    elif page == "Data":
        data_section()
    elif page == "Equations":
        equations_section()
    elif page == "Results":
        results_section()

# ---------------------------------------------------------------------
# Existing sections (Overview, Methodology, Regions, Data, Equations, Results)
# ---------------------------------------------------------------------

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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">System Relationships</div>', unsafe_allow_html=True)
    st.markdown("*(Placeholder for your diagram—replace with an actual image URL if desired.)*")
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

# ---------------------------------------------------------------------
# NEW SECTION: System Formation
# ---------------------------------------------------------------------
def system_formation():
    st.markdown('<div class="section-title">System Formation</div>', unsafe_allow_html=True)

    # Overview Card
    st.markdown(
        """
        <div class="card">
            <div class="card-title">System Development Process</div>
            <p>Our nonlinear dynamical system was developed through a systematic approach 
            considering the key interactions between environmental, economic, and operational factors 
            in semiconductor manufacturing.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # State Variables Card
    st.markdown(
        """
        <div class="card bg-blue-50">
            <div class="card-title text-blue-800">State Variables</div>
            <ul class="card-list">
                <li><strong>P(t): Production Capacity</strong><br/>
                    • Represents manufacturing output capability<br/>
                    • Influenced by resource availability and demand</li>
                <li><strong>W(t): Water Availability</strong><br/>
                    • Measures water resources accessible for production<br/>
                    • Affected by regional conditions and recycling</li>
                <li><strong>E(t): Energy Availability</strong><br/>
                    • Represents power supply stability<br/>
                    • Includes renewable and traditional sources</li>
                <li><strong>C(t): Compliance Level</strong><br/>
                    • Environmental regulation adherence<br/>
                    • Affects operational constraints</li>
                <li><strong>R(t): Resource Efficiency</strong><br/>
                    • Measures resource utilization effectiveness<br/>
                    • Influences sustainability metrics</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # System Interactions Card
    st.markdown(
        """
        <div class="card bg-green-50">
            <div class="card-title text-green-800">Key System Interactions</div>
            <ul class="card-list">
                <li><strong>Production-Resource Coupling</strong><br/>
                    • Nonlinear relationship between production and resource consumption<br/>
                    • Capacity constraints and efficiency factors</li>
                <li><strong>Environmental Feedback</strong><br/>
                    • Resource availability affects production capabilities<br/>
                    • Environmental conditions influence efficiency</li>
                <li><strong>Regulatory Impact</strong><br/>
                    • Compliance requirements affect operational parameters<br/>
                    • Policy changes create system perturbations</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Equation Formation Card
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Equation Development</div>
            <p>Our system equations incorporate:</p>
            <ul class="card-list">
                <li><strong>Logistic Growth Terms:</strong> (1 - P/K) for capacity constraints</li>
                <li><strong>Quadratic Damping:</strong> -δ₁D(t)P² for resource limitations</li>
                <li><strong>Cubic Interactions:</strong> λ₁(C*(t) - C(t))³ for compliance effects</li>
                <li><strong>Resource Coupling:</strong> E(t)W(t) for environmental interactions</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mermaid Diagram (as a code block). 
    # NOTE: By default, Streamlit won't "render" Mermaid syntax. 
    # This will just show as text unless you use a custom component or extension. 
    # Alternatively, embed an SVG from mermaid.ink. For demonstration, we show the code:
    st.markdown("**System Interaction Diagram (Mermaid)**")
    st.markdown(
        """
        ```mermaid
        graph TD
            subgraph Production System
                P[Production Capacity]
                M[Manufacturing Rate]
                D[Demand]
            end

            subgraph Environmental Resources
                W[Water Availability]
                E[Energy Supply]
                R[Resource Efficiency]
            end

            subgraph Regulatory Framework
                C[Compliance Level]
                REG[Environmental Regulations]
            end

            P -- Consumes --> W
            P -- Requires --> E
            W -- Constrains --> P
            E -- Powers --> P
            R -- Improves --> W
            R -- Optimizes --> E
            C -- Controls --> R
            REG -- Defines --> C
            P -- Impacts --> C

            style P fill:#bae6fd,stroke:#0284c7
            style W fill:#bbf7d0,stroke:#16a34a
            style E fill:#bbf7d0,stroke:#16a34a
            style R fill:#bbf7d0,stroke:#16a34a
            style C fill:#fde68a,stroke:#d97706
            style REG fill:#fde68a,stroke:#d97706
            style M fill:#bae6fd,stroke:#0284c7
            style D fill:#bae6fd,stroke:#0284c7
        ```
        """,
        unsafe_allow_html=True
    )

def regions_section():
    st.markdown('<div class="section-title">Regional Analysis</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="card bg-green-50">', unsafe_allow_html=True)
    st.markdown('<div class="card-title text-green-800">Easily Accessible Data</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <ul class="card-list">
            <li><strong>Regional Energy Consumption</strong><br/><em>Source: DOE</em></li>
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
        <p>Below is a conceptual overview of a nonlinear dynamical system describing 
        production (P), water (W), energy (E), etc. in ASCII-friendly notation:</p>
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

if __name__ == "__main__":
    main()
