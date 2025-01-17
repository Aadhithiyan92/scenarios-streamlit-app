import streamlit as st

def main():
    # Set page title
    st.set_page_config(page_title="Environmental-Economic Sensitivity Analysis", layout="wide")

    # Top Bar / Header
    st.markdown(
        """
        <div style="background: linear-gradient(to right, #2563EB, #1E40AF); padding: 20px; border-radius: 5px;">
            <h1 style="color: white; margin-bottom: 0;">Environmental-Economic Sensitivity Analysis</h1>
            <p style="color: #DDEAF7;">US Regional Semiconductor Supply Chain Dynamics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar menu
    menu = ["Overview", "Methodology", "Regions", "Data", "Equations", "Results"]
    choice = st.sidebar.radio("Sections", menu)

    # Render the appropriate section based on selection
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
    st.subheader("Research Overview")
    st.write(
        """
        This cutting-edge research investigates the complex dynamics between environmental 
        sustainability and semiconductor supply chain resilience across key US manufacturing regions. 
        Using advanced nonlinear dynamical systems analysis, we model the intricate relationships 
        between water availability, energy transitions, and environmental regulations.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Key Innovations")
        st.markdown("- Nonlinear coupling of environmental-economic factors")
        st.markdown("- Regional sensitivity analysis framework")
        st.markdown("- Critical threshold identification methods")

    with col2:
        st.markdown("### Expected Impact")
        st.markdown("- Enhanced supply chain resilience strategies")
        st.markdown("- Regional policy recommendations")
        st.markdown("- Sustainability-oriented manufacturing practices")


def regions_section():
    st.subheader("Regional Analysis")
    st.write("Key manufacturing regions and their critical environmental-economic characteristics:")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Southwest Region (AZ, NM)")
        st.write(
            "- Water scarcity challenges\n"
            "- High solar energy potential\n"
            "- Major players: Intel, TSMC"
        )

        st.markdown("---")

        st.markdown("#### Texas Region")
        st.write(
            "- Independent power grid (ERCOT)\n"
            "- Mixed energy sources\n"
            "- Major players: Samsung, TI"
        )

    with col2:
        st.markdown("#### Pacific Northwest (OR, WA)")
        st.write(
            "- Hydroelectric power availability\n"
            "- Stable water supply\n"
            "- Major player: Intel"
        )

        st.markdown("---")

        st.markdown("#### Northeast Corridor (NY)")
        st.write(
            "- Stable water resources\n"
            "- Strict environmental regulations\n"
            "- Major player: GlobalFoundries"
        )


def data_section():
    st.subheader("Data Requirements & Accessibility")

    st.write("Below is an overview of easily accessible, moderately difficult, and challenging data sources.")

    # We'll use columns to replicate a 3-column layout:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Easily Accessible Data")
        st.markdown("- **Regional Energy Consumption** (DOE)")
        st.markdown("- **Water Usage Permits** (State Environmental Agencies)")
        st.markdown("- **Environmental Compliance Records** (EPA Database)")

    with col2:
        st.markdown("### Moderately Difficult Data")
        st.markdown("- **Facility Energy Usage** (Company Reports)")
        st.markdown("- **Water Recycling Rates** (Industry Surveys)")
        st.markdown("- **Production Capacity** (Industry Reports)")

    with col3:
        st.markdown("### Challenging Data")
        st.markdown("- **Efficiency Metrics** (Proprietary Data)")
        st.markdown("- **Production Costs** (Internal Records)")
        st.markdown("- **Environmental Targets** (Corporate Plans)")


def methodology_section():
    st.subheader("Research Methodology")

    # System Relationships
    st.markdown("#### System Relationships")
    st.markdown(
        """
        *Using a high-level diagram to illustrate water, energy, and production interdependencies.*
        """
    )
    st.image("https://example.com/system-relationships-diagram.svg", use_column_width=True)

    # Two-column layout for analysis methods and implementation steps
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Analysis Methods")
        st.markdown("- **Lyapunov Stability Analysis**: Examining system stability near equilibrium points")
        st.markdown("- **Bifurcation Analysis**: Identifying critical parameter thresholds")
        st.markdown("- **Sensitivity Analysis**: Parameter impact assessment")

    with col2:
        st.markdown("### Implementation Steps")
        st.markdown("- **Data Collection & Validation**: Regional environmental and production data")
        st.markdown("- **Model Calibration**: Parameter estimation and validation")
        st.markdown("- **Regional Analysis**: Comparative regional studies")


def equations_section():
    st.subheader("System Equations")

    st.markdown("#### Nonlinear Dynamical System (Conceptual Overview)")
    st.code(
        """
# Example (ASCII-friendly) version:
# dP/dt = mu1 * M(t)*E(t)*W(t)*(1 - P/K) - delta1 * D(t)*P^2
# dW/dt = alpha2 * P(t) * (1 - W/Wmax) - beta2 * R(t)*W^2 - delta2 * T(t)
# dE/dt = [alpha1 * P(t) + beta1 * M(t)](1 - E/Emax) - gamma1 * R(t)*E^2
# ...
        """,
        language="python",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Stability Analysis")
        st.markdown("- Lyapunov stability analysis")
        st.markdown("- Bifurcation analysis")
        st.markdown("- Phase space analysis")

    with col2:
        st.markdown("### Parameters")
        st.markdown("- Environmental coupling coefficients")
        st.markdown("- Production efficiency factors")
        st.markdown("- Resource utilization rates")


def results_section():
    st.subheader("Expected Results & Impact")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Research Outcomes")
        st.markdown("- **Regional Stability Maps**: Identification of stability boundaries/thresholds")
        st.markdown("- **Sensitivity Metrics**: System response to parameter variations")
        st.markdown("- **Risk Assessment Framework**: Evaluation of environmental risks to supply chain")

    with col2:
        st.markdown("### Expected Impact")
        st.markdown("- **Policy Recommendations**: Guidance for regional policymakers")
        st.markdown("- **Industry Guidelines**: Best practices for environmental sustainability")
        st.markdown("- **Resilience Strategies**: Actionable plans for supply chain strengthening")


if __name__ == "__main__":
    main()
