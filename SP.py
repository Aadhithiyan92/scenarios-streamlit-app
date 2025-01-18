import streamlit as st

def main():
    st.title("Environmental-Economic Sensitivity Analysis")
    st.subheader("US Regional Semiconductor Supply Chain Dynamics")

    # 1) Define pages in a list (including "System Formation").
    pages = [
        "Overview", 
        "Objectives", 
        "Methodology", 
        "System Formation",  # <--- NEW PAGE
        "Data Requirements", 
        "Regional Analysis", 
        "Results"
    ]

    # 2) Create a selectbox for the user to pick a page.
    section = st.selectbox("Select Section", pages)

    # 3) Show the appropriate page’s content.
    if section == "Overview":
        overview_section()
    elif section == "Objectives":
        objectives_section()
    elif section == "Methodology":
        methodology_section()
    elif section == "System Formation":
        system_formation()  # <--- NEW PAGE FUNCTION
    elif section == "Data Requirements":
        data_requirements_section()
    elif section == "Regional Analysis":
        regional_analysis_section()
    elif section == "Results":
        results_section()

def overview_section():
    st.header("Overview")
    st.write(
        """
        This cutting-edge research investigates the complex dynamics between environmental 
        sustainability and semiconductor supply chain resilience across key US manufacturing regions.
        """
    )

def objectives_section():
    st.header("Objectives")
    st.write(
        """
        1. Analyze environmental constraints affecting semiconductor production.
        2. Identify critical thresholds for regional resource sustainability.
        3. Provide policy recommendations to improve supply chain resilience.
        """
    )

def methodology_section():
    st.header("Methodology")
    st.write(
        """
        We employ advanced system dynamics and nonlinear analysis techniques to model 
        interdependencies among water availability, energy consumption, and production throughput 
        in semiconductor manufacturing across multiple US regions.
        """
    )

def system_formation():
    """
    Your new "System Formation" content, as provided in your snippet.
    """
    st.header("System Formation")

    # Overview Card
    st.markdown(
        """
        <div style="background-color: #FFF; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h3 style="color:#1E3A8A; margin-bottom:0.5rem;">System Development Process</h3>
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
        <div style="background-color: #EFF6FF; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h3 style="color:#1E3A8A; margin-bottom:0.5rem;">State Variables</h3>
            <ul style="list-style-type: disc; padding-left:1.5rem;">
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
        <div style="background-color: #ECFDF5; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h3 style="color:#065F46; margin-bottom:0.5rem;">Key System Interactions</h3>
            <ul style="list-style-type: disc; padding-left:1.5rem;">
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
        <div style="background-color: #FFF; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h3 style="color:#1E3A8A; margin-bottom:0.5rem;">Equation Development</h3>
            <p>Our system equations incorporate:</p>
            <ul style="list-style-type: disc; padding-left:1.5rem;">
                <li><strong>Logistic Growth Terms:</strong> (1 - P/K) for capacity constraints</li>
                <li><strong>Quadratic Damping:</strong> -δ₁D(t)P² for resource limitations</li>
                <li><strong>Cubic Interactions:</strong> λ₁(C*(t) - C(t))³ for compliance effects</li>
                <li><strong>Resource Coupling:</strong> E(t)W(t) for environmental interactions</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mermaid Diagram Code (will appear as text unless using a mermaid component or an SVG link)
    st.markdown("**System Interaction Diagram (Mermaid Code):**")
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

def data_requirements_section():
    st.header("Data Requirements")
    st.write(
        """
        - **Easily Accessible Data**: Regional Energy Consumption, Water Usage Permits, etc.
        - **Moderately Difficult Data**: Facility Energy Usage, Water Recycling Rates, etc.
        - **Challenging Data**: Efficiency Metrics, Production Costs, etc.
        """
    )

def regional_analysis_section():
    st.header("Regional Analysis")
    st.write(
        """
        1. **Southwest Region (AZ, NM)** 
           - Water scarcity challenges, high solar potential, major players: Intel, TSMC
        2. **Pacific Northwest (OR, WA)**
           - Hydroelectric power, stable water supply, major player: Intel
        3. **Texas Region**
           - Independent power grid (ERCOT), mixed energy sources, Samsung, TI
        4. **Northeast Corridor (NY)**
           - Stable water resources, strict regulations, major player: GlobalFoundries
        """
    )

def results_section():
    st.header("Results & Impact")
    st.write(
        """
        - **Regional Stability Maps**: Identifying stability boundaries
        - **Sensitivity Metrics**: Measuring system response to parameter variations
        - **Risk Assessment**: Evaluating environmental risks to supply chain stability
        - **Policy Recommendations**: Guidance for policymakers
        - **Industry Guidelines**: Best practices for sustainability
        - **Resilience Strategies**: Strengthening the semiconductor supply chain
        """
    )

if __name__ == "__main__":
    main()
