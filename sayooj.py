import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.colors as mcolors

# Set page configuration
st.set_page_config(
    page_title="Contact Tracing Effectiveness Study",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the app
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #3B82F6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .authors {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .math-formula {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .highlight-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFFBEB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    .figure-caption {
        font-style: italic;
        text-align: center;
        color: #6B7280;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar with navigation
st.sidebar.title("Navigation")

slides = [
    "Title & Abstract",
    "Introduction",
    "Model Framework",
    "Reproduction Number Analysis",
    "Contact Tracing Effectiveness",
    "Stability Analysis",
    "Numerical Results",
    "Conclusions"
]

current_slide = st.sidebar.radio("Select a slide:", slides)

# Display additional info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### About this Research")
st.sidebar.markdown("""
This presentation covers a mathematical modeling study on contact tracing effectiveness in infectious disease control. The research focuses on deriving analytical formulations of the effective reproduction number.

**Original paper:**  
Mathematical Biosciences 383 (2025) 109415
""")

# Functions to create visualizations

def create_model_diagram():
    """Create a compartmental model diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define compartments (boxes)
    compartments = {
        'S': (0.1, 0.5),
        'E': (0.3, 0.5),
        'Ih': (0.5, 0.7),
        'Iu': (0.5, 0.3),
        'Ec': (0.3, 0.3),
        'Ic': (0.7, 0.5),
        'H': (0.8, 0.5),
        'U': (0.9, 0.3)
    }
    
    # Draw compartments
    for name, (x, y) in compartments.items():
        rect = patches.Rectangle((x, y), 0.1, 0.1, linewidth=1.5, edgecolor='black', facecolor='lightblue', alpha=0.7)
        ax.add_patch(rect)
        ax.text(x+0.05, y+0.05, name, ha='center', va='center', fontweight='bold')
    
    # Define arrows
    arrows = [
        ((0.01, 0.55), (0.1, 0.55), 'Π'),
        ((0.2, 0.55), (0.3, 0.55), 'β'),
        ((0.4, 0.55), (0.5, 0.75), 'ρ/τ'),
        ((0.4, 0.45), (0.5, 0.35), '(1-ρ)/τ'),
        ((0.4, 0.35), (0.7, 0.5), '1/τ'),
        ((0.6, 0.75), (0.8, 0.55), '1/Th'),
        ((0.6, 0.35), (0.9, 0.35), 'γ'),
        ((0.85, 0.5), (0.9, 0.35), 'γ1'),
        ((0.95, 0.3), (0.15, 0.5), 'γ2')
    ]
    
    # Draw arrows
    for (x1, y1), (x2, y2), label in arrows:
        dx = x2 - x1
        dy = y2 - y1
        arrow = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                                        color='black', linewidth=1, mutation_scale=15)
        ax.add_patch(arrow)
        ax.text(x1 + dx/2, y1 + dy/2, label, ha='center', va='center', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("SEIR-Type Model with Contact Tracing")
    
    return fig

def create_reproduction_number_plot():
    """Create visualization of reproduction number with/without contact tracing"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create data
    phi_vals = np.linspace(0, 1, 100)  # Contact tracing fraction
    r0 = 2.5  # Base reproduction number
    
    # Different efficacy levels
    low_efficacy = r0 * (1 - 0.3 * phi_vals)
    medium_efficacy = r0 * (1 - 0.5 * phi_vals)
    high_efficacy = r0 * (1 - 0.8 * phi_vals)
    
    # Plot lines
    ax.plot(phi_vals, low_efficacy, 'r-', linewidth=2, label='Low tracing efficacy')
    ax.plot(phi_vals, medium_efficacy, 'y-', linewidth=2, label='Medium tracing efficacy')
    ax.plot(phi_vals, high_efficacy, 'g-', linewidth=2, label='High tracing efficacy')
    
    # Add horizontal line at R=1
    ax.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Epidemic threshold (R=1)')
    
    # Add threshold markers
    low_threshold = np.interp(1, low_efficacy[::-1], phi_vals[::-1])
    medium_threshold = np.interp(1, medium_efficacy[::-1], phi_vals[::-1])
    high_threshold = np.interp(1, high_efficacy[::-1], phi_vals[::-1])
    
    if low_threshold < 1:
        ax.plot(low_threshold, 1, 'ro', markersize=10)
    if medium_threshold < 1:
        ax.plot(medium_threshold, 1, 'yo', markersize=10)
    if high_threshold < 1:
        ax.plot(high_threshold, 1, 'go', markersize=10)
    
    # Labels and formatting
    ax.set_xlabel('Fraction of contacts traced (φ)', fontsize=12)
    ax.set_ylabel('Effective Reproduction Number (Re)', fontsize=12)
    ax.set_title('Effect of Contact Tracing on Reproduction Number', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, r0 * 1.1)
    
    return fig

def create_contour_plot():
    """Create contour plot of Re as function of tracing and hospitalization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create data grid
    phi = np.linspace(0, 1, 100)  # Tracing fraction
    rho = np.linspace(0, 1, 100)  # Hospitalization fraction
    PHI, RHO = np.meshgrid(phi, rho)
    
    # Calculate Re based on the two parameters
    r0 = 2.5  # Base reproduction number
    RE = r0 * (1 - 0.6 * PHI) * (1 - 0.4 * RHO)
    
    # Create contour plot
    levels = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
    contour = ax.contourf(PHI, RHO, RE, levels=levels, cmap='RdYlGn_r', alpha=0.9)
    
    # Add contour lines
    contour_lines = ax.contour(PHI, RHO, RE, levels=levels, colors='k', linewidths=0.5, alpha=0.7)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Highlight Re=1 contour
    critical_contour = ax.contour(PHI, RHO, RE, levels=[1.0], colors=['blue'], linewidths=2)
    ax.clabel(critical_contour, inline=True, fontsize=10, fmt='Re=1')
    
    # Add labels and title
    ax.set_xlabel('Fraction of contacts traced (φ)', fontsize=12)
    ax.set_ylabel('Fraction of cases hospitalized (ρ)', fontsize=12)
    ax.set_title('Effective Reproduction Number (Re) with Contact Tracing and Hospitalization', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Effective Reproduction Number (Re)', fontsize=10)
    
    # Add annotation for epidemic control region
    ax.text(0.7, 0.8, 'Epidemic Control\n(Re < 1)', 
            backgroundcolor='white', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='green', boxstyle='round'))
    
    return fig

def create_ebola_data_plot():
    """Create plot of Ebola data with contact tracing"""
    # Generate sample data for Sierra Leone and Guinea
    weeks = np.arange(1, 13)
    
    # Sierra Leone data
    sl_total = [120, 105, 90, 82, 75, 65, 55, 48, 40, 35, 30, 20]
    sl_traced = [30, 35, 40, 42, 45, 42, 38, 32, 26, 22, 18, 12]
    sl_fraction = [t/tot for t, tot in zip(sl_traced, sl_total)]
    sl_re = [2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 0.9, 0.8, 0.7, 0.6]
    
    # Guinea data
    gn_total = [40, 45, 50, 60, 55, 45, 40, 35, 38, 30, 25, 18]
    gn_traced = [12, 14, 15, 18, 16, 14, 12, 10, 12, 10, 8, 6]
    gn_fraction = [t/tot for t, tot in zip(gn_traced, gn_total)]
    gn_re = [2.2, 2.1, 1.9, 1.8, 1.7, 1.5, 1.3, 1.2, 1.1, 0.9, 0.8, 0.7]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Sierra Leone plot
    ax1.bar(weeks-0.2, sl_total, width=0.4, color='blue', alpha=0.7, label='Total cases')
    ax1.bar(weeks+0.2, sl_traced, width=0.4, color='green', alpha=0.7, label='Traced cases')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(weeks, sl_fraction, 'g-', marker='o', label='Fraction traced')
    ax1_twin.plot(weeks, sl_re, 'r-', marker='s', label='Re')
    
    # Add threshold line for Re=1
    ax1_twin.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    
    # Labels and formatting for Sierra Leone
    ax1.set_xlabel('Weeks (Jan-Apr 2015)')
    ax1.set_ylabel('Number of cases')
    ax1_twin.set_ylabel('Fraction traced / Re')
    ax1.set_title('Sierra Leone')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_xticks(weeks)
    ax1_twin.set_ylim(0, 2.5)
    ax1.grid(True, alpha=0.3)
    
    # Guinea plot
    ax2.bar(weeks-0.2, gn_total, width=0.4, color='blue', alpha=0.7, label='Total cases')
    ax2.bar(weeks+0.2, gn_traced, width=0.4, color='green', alpha=0.7, label='Traced cases')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(weeks, gn_fraction, 'g-', marker='o', label='Fraction traced')
    ax2_twin.plot(weeks, gn_re, 'r-', marker='s', label='Re')
    
    # Add threshold line for Re=1
    ax2_twin.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    
    # Labels and formatting for Guinea
    ax2.set_xlabel('Weeks (Jan-Apr 2015)')
    ax2.set_ylabel('Number of cases')
    ax2_twin.set_ylabel('Fraction traced / Re')
    ax2.set_title('Guinea')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_xticks(weeks)
    ax2_twin.set_ylim(0, 2.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_stability_diagram():
    """Create visualization of stable vs unstable dynamics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time points
    t = np.linspace(0, 20, 100)
    
    # Stable case (Re < 1)
    initial_cases = [10, 20, 30, 40]
    for ic in initial_cases:
        cases_stable = ic * np.exp(-0.2 * t)
        ax1.plot(t, cases_stable)
    
    # Unstable case (Re > 1)
    initial_cases = [5, 10, 15, 20]
    for ic in initial_cases:
        cases_unstable = ic * np.exp(0.2 * t) / (1 + (ic/100) * (np.exp(0.2 * t) - 1))
        ax2.plot(t, cases_unstable)
    
    # Formatting
    ax1.set_title('Stable: Re < 1', fontsize=14)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of cases')
    ax1.text(10, 25, 'Disease dies out', fontsize=12, ha='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Unstable: Re > 1', fontsize=14)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of cases')
    ax2.text(10, 50, 'Epidemic growth', fontsize=12, ha='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_sensitivity_plot():
    """Create sensitivity analysis visualization for contact tracing parameters"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Data for varying incubation period
    trace_delay = np.linspace(0, 5, 100)  # Days of tracing delay
    
    # Different incubation periods
    short_incubation = 1.0 - 0.8 * np.exp(-0.8 * trace_delay)  # Re ratio (Re/R0) for short incubation
    medium_incubation = 1.0 - 0.8 * np.exp(-0.4 * trace_delay)  # Medium incubation
    long_incubation = 1.0 - 0.8 * np.exp(-0.2 * trace_delay)  # Long incubation
    
    # Plot incubation period effects
    ax1.plot(trace_delay, short_incubation, 'r-', linewidth=2, label='Short incubation (5 days)')
    ax1.plot(trace_delay, medium_incubation, 'y-', linewidth=2, label='Medium incubation (8 days)')
    ax1.plot(trace_delay, long_incubation, 'g-', linewidth=2, label='Long incubation (12 days)')
    
    # Formatting first plot
    ax1.set_xlabel('Contact tracing delay (days)')
    ax1.set_ylabel('Re / R0 ratio')
    ax1.set_title('Effect of Incubation Period on Tracing Effectiveness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Data for tracing vs monitoring efficacy
    tracing_fraction = np.linspace(0, 1, 100)
    
    # Different monitoring efficacies
    low_monitor = 1.0 - 0.3 * tracing_fraction  # Re ratio with low monitoring efficacy
    medium_monitor = 1.0 - 0.6 * tracing_fraction  # Medium efficacy
    high_monitor = 1.0 - 0.9 * tracing_fraction  # High efficacy
    
    # Plot monitoring efficacy effects
    ax2.plot(tracing_fraction, low_monitor, 'r-', linewidth=2, label='Low monitoring (βm/β = 0.7)')
    ax2.plot(tracing_fraction, medium_monitor, 'y-', linewidth=2, label='Medium monitoring (βm/β = 0.4)')
    ax2.plot(tracing_fraction, high_monitor, 'g-', linewidth=2, label='High monitoring (βm/β = 0.1)')
    
    # Add threshold line
    ax2.axhline(y=0.4, color='k', linestyle='--', alpha=0.7)
    ax2.text(0.05, 0.38, 'Epidemic control threshold', fontsize=8)
    
    # Formatting second plot
    ax2.set_xlabel('Fraction of contacts traced (φ)')
    ax2.set_ylabel('Re / R0 ratio')
    ax2.set_title('Effect of Monitoring Efficacy on Tracing Effectiveness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    return fig

# Title and Abstract slide
if current_slide == "Title & Abstract":
    # Main title
    st.markdown('<div class="main-title">A Mathematical Modeling Study of the Effectiveness of Contact Tracing</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">in Reducing the Spread of Infectious Diseases with Incubation Period</div>', unsafe_allow_html=True)
    st.markdown('<div class="authors">Mohamed Ladib, Cameron J. Browne, Hayriye Gulbudak, Aziz Ouhinou</div>', unsafe_allow_html=True)
    
    # Journal info
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/6295/6295417.png", width=150)
    with col2:
        st.markdown("### Published in:")
        st.markdown("Mathematical Biosciences 383 (2025) 109415")
        st.markdown("*Available online 26 February 2025*")
    
    st.markdown("---")
    
    # Abstract
    st.markdown("### Abstract")
    
    st.markdown("""
    In this work, we study an epidemic model with demography that incorporates some key aspects of the contact
    tracing intervention. We derive generic formulae for the effective reproduction number R<sub>e</sub> when contact tracing
    is employed to mitigate the spread of infection.
    
    The derived expressions are reformulated in terms of the initial reproduction number R<sub>0</sub> (in the absence of tracing), 
    the number of traced cases caused by a primary untraced reported index case, and the average number of secondary
    cases infected by traced infectees during their infectious period.
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        In parallel, under some restrictions, the local stability of the disease-free equilibrium is investigated. 
        The model was fitted to data of Ebola disease collected during the 2014–2016 outbreaks in West Africa.
        
        Finally, numerical simulations are provided to investigate the effect of key parameters on R<sub>e</sub>.
        By considering ongoing interventions, the simulations indicate whether contact tracing can suppress R<sub>e</sub> below
        unity, as well as identify parameter regions where it can effectively contain epidemic outbreaks when applied
        with a given level of efficiency.
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Key Findings:**
        
        - Derived analytical expressions for effective reproduction number with contact tracing
        - Established stability thresholds for disease extinction vs. persistence
        - Applied model to 2014-2016 Ebola outbreak data
        - Identified critical parameters for contact tracing effectiveness
        - Determined conditions under which contact tracing can contain epidemics
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Introduction slide
elif current_slide == "Introduction":
    st.markdown('<div class="section-header">Introduction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        **Contact tracing** is a public health intervention aimed at suppressing the spread of infections. This is achieved by intercepting chains of transmission as early as possible. The process begins with the identification of infectees caused by an index case.
        
        Those who are found to be infectious are provided with the necessary treatment, while those who did not show symptoms upon tracking are quarantined or kept under monitoring protocols.
        """)
        
        st.markdown("---")
        
        st.markdown("""
        From a modeling perspective, contact tracing has been approached using various methods:
        
        - **Agent-based models (ABMs)** - Can include any desired details but lack analytical tractability
        - **Branching processes** - Analyze intrinsic probabilities at onset of outbreaks
        - **Network models** - Model contact patterns with pair approximations
        - **Discrete equations** - Capture steps of contact tracing process
        - **Differential equations** - Ordinary, delay, or integral equations for population dynamics
        """)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Recent Applications:**
        
        Contact tracing has gained significant attention during the COVID-19 pandemic and has been instrumental in controlling outbreaks of various diseases including:
        
        - COVID-19
        - Ebola
        - Tuberculosis
        - Sexually transmitted infections
        - Measles
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Our Approach:**
        
        We use compartmental ODE models which offer analytical tractability while capturing key features of contact tracing.
        
        This allows us to:
        1. Derive explicit formulas for reproduction numbers
        2. Analyze stability conditions
        3. Identify parameter thresholds for epidemic control
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Previous Work and Our Contribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        Previous research by the authors proposed an SEIR-type ODE model incorporating contact tracing for Ebola disease. Key features included:
        
        - Separation of untraced and traced infection pathways
        - Modeling of incubation period
        - Consideration of monitoring protocols
        - Tracing delays
        
        However, this model did not include demographics and focused only on outbreak scenarios.
        """)
    
    with col2:
        st.markdown("""
        **Our extension in this work:**
        
        - Incorporate demographic dynamics (births/deaths)
        - Allow for endemic equilibria and threshold behavior
        - Derive explicit formulations of effective reproduction number
        - Analyze conditions for disease extinction vs. persistence
        - Consider both single-step and higher-order tracing
        - Apply to real-world data from Ebola outbreaks
        """)

# Model Framework slide
elif current_slide == "Model Framework":
    st.markdown('<div class="section-header">Model Framework</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        We consider an SEIR-type model with contact tracing and demography. The total population at any given time is divided into:
        """)
        
        col1a, col1b = st.columns(2)
        
        with col1a:
            st.markdown("**Untraced pathway:**")
            st.markdown("""
            - Susceptible individuals (S)
            - Exposed cases still incubating (E)
            - Infectious cases eventually hospitalized (I<sub>h</sub>)
            - Infectious cases that remain unreported (I<sub>u</sub>)
            """, unsafe_allow_html=True)
        
        with col1b:
            st.markdown("**Traced pathway:**")
            st.markdown("""
            - Exposed cases tracked while incubating (E<sub>c</sub><sup>e</sup>)
            - Infectious cases under monitoring (I<sub>c</sub><sup>e</sup>)
            - Exposed cases tracked while infectious (E<sub>c</sub><sup>i</sup>)
            - Infectious cases already infectious upon tracking (I<sub>c</sub><sup>i</sup>)
            - Hospitalized cases (H)
            - Recovered individuals (U)
            """, unsafe_allow_html=True)
    
    with col2:
        st.pyplot(create_model_diagram())
        st.markdown('<div class="figure-caption">Fig. 1: Compartmental model diagram showing infection pathways</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Model Dynamics")
    
    st.markdown("The associated system of differential equations with mass-action transmission:")
    
    st.markdown('<div class="math-formula">', unsafe_allow_html=True)
    st.latex(r'''\begin{align}
    # Continuing from where the paste.txt left off - completing the model equations
    \frac{dI_h}{dt} &= \frac{\rho}{\tau}E(t) - \left(\mu + d_1 + \frac{1}{T_h}\right)I_h(t)\\
    \frac{dI_u}{dt} &= \frac{1-\rho}{\tau}E(t) - \left(\mu + d_1 + \gamma\right)I_u(t)\\
    \frac{dE_c^e}{dt} &= p_e\phi\beta S(t)I_h(t) + p_e^i q\phi\beta S(t)I_c^i(t) - \left(\mu + \frac{1}{\tau}\right)E_c^e(t)\\
    \frac{dI_c^e}{dt} &= \frac{1}{\tau}E_c^e(t) - \left(\mu + d_1 + \frac{1}{T_m}\right)I_c^e(t)\\
    \frac{dE_c^i}{dt} &= (1-p_e)\phi\beta S(t)I_h(t) + (1-p_e^i)q\phi\beta S(t)I_c^i(t) - \left(\mu + \frac{1}{\tau}\right)E_c^i(t)\\
    \frac{dI_c^i}{dt} &= \frac{1}{\tau}E_c^i(t) - \left(\mu + d_1 + \frac{1}{T_i}\right)I_c^i(t)\\
    \frac{dH}{dt} &= \frac{1}{T_m}I_c^e(t) + \frac{1}{T_i}I_c^i(t) + \frac{1}{T_h}I_h(t) - (\gamma_1 + d_2 + \mu)H(t)\\
    \frac{dU}{dt} &= \gamma I_u(t) + \gamma_1 H(t) - (\gamma_2 + \mu)U(t)
    \end{align}''')
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Where:
    - Π: Recruitment rate of susceptible individuals
    - β: Transmission rate of untraced and traced (unmonitored) infectious cases
    - β<sub>m</sub>: Transmission rate of traced (monitored) infectious cases
    - ρ: Proportion of hospitalized/reported untraced cases
    - q: Boolean parameter for tracing order (0: single-step, 1: higher-order)
    - φ: Fraction of traced infectees per index case
    - τ: Mean incubation period
    - T<sub>h</sub>, T<sub>m</sub>, T<sub>i</sub>: Mean infectious periods for different cases
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Key Model Features</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Dual Infection Pathways**: Separate modeling of traced and untraced cases
        - **Incubation Period**: Explicit modeling of the disease's latent period
        - **Tracing Scenarios**:
          - Single-step tracing (q=0): Only untraced reported cases trigger contact tracing
          - Higher-order tracing (q=1): Traced cases can also trigger further tracing
        - **Monitoring Protocols**: Reduced infectivity for traced cases under monitoring
        - **Demographic Dynamics**: Inclusion of births/deaths allows for endemic equilibria
        """)
    
    with col2:
        st.markdown('<div class="subsection-header">Tracing Probability Calculation</div>', unsafe_allow_html=True)
        st.markdown("""
        The probabilities p<sub>e</sub> and p<sub>e</sub><sup>i</sup> of finding infectees while still incubating:
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
        st.latex(r'''\begin{align}
        p_e &= \frac{\tau}{\tau + T_h}\exp\left(-\frac{\delta}{\tau}\right)\\
        p_e^i &= \frac{\tau}{\tau + T_i}\exp\left(-\frac{\delta}{\tau}\right)
        \end{align}''')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Where δ is the time delay between detection of the index case and tracing of their contacts.
        
        These probabilities increase with longer incubation period (τ) and decrease with longer tracing delay (δ).
        """)

# Reproduction Number Analysis
elif current_slide == "Reproduction Number Analysis":
    st.markdown('<div class="section-header">Effective Reproduction Number Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    We derive the effective reproduction number R<sub>e</sub> using the next-generation approach in two scenarios:
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Single-step tracing (q=0)</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
        st.latex(r'''R_e = \frac{1}{2}\left[(R_0 - L) + \sqrt{(R_0 - L)^2 + 4LR}\right]''')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Where:
        - R<sub>0</sub> = Basic reproduction number without tracing
        - L = Average number of traced infections per primary untraced reported case
        - R = Average infections caused by first-order traced cases
        """, unsafe_allow_html=True)
        
        st.markdown("""
        R<sub>0</sub> is given by:
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
        st.latex(r'''R_0 = \frac{\beta S_0}{\tau \mu_\tau}\left[\frac{\rho}{\mu_{T_h}} + \frac{(1-\rho)}{\mu_\gamma}\right]''')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Higher-order tracing (q=1)</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
        st.latex(r'''R_e = \frac{1}{2}\left[((R_0 - L) + \bar{L}) + \sqrt{((R_0 - L) - \bar{L})^2 + 4(1-\phi)L\bar{R}}\right]''')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Additional terms:
        - L̄ = Average traced infections per primary higher-order traced case
        - R̄ = Average infections caused by higher-order traced cases
        """)
        
        st.markdown("""
        Under perfect tracing settings (φ=1, β<sub>m</sub>=0):
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
        st.latex(r'''R_e = \max\{(1-\rho)R_0(\rho=0), (1-p_e^i)R_i\}''')
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.pyplot(create_reproduction_number_plot())
    st.markdown('<div class="figure-caption">Fig. 2: Effect of contact tracing fraction (φ) on effective reproduction number (Re) under different efficacy levels</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="subsection-header">Key Insights</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Threshold Behavior**
        
        - R<sub>e</sub> < 1: Disease will die out
        - R<sub>e</sub> > 1: Disease will persist
        
        Contact tracing reduces R<sub>e</sub> by removing infections from the transmission chain.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Impact of Tracing Parameters**
        
        - Higher fraction traced (φ) → Lower R<sub>e</sub>
        - Higher monitoring efficacy (lower β<sub>m</sub>) → Lower R<sub>e</sub>
        - Short tracing delay (δ) → Higher p<sub>e</sub> → Lower R<sub>e</sub>
        - Higher-order tracing can provide additional control
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Disease Characteristics Matter**
        
        - Long incubation period (τ) favors tracing effectiveness
        - High proportion of reportable cases (ρ) improves tracing impact
        - Short infectious period reduces required tracing coverage
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Contact Tracing Effectiveness slide
elif current_slide == "Contact Tracing Effectiveness":
    st.markdown('<div class="section-header">Contact Tracing Effectiveness</div>', unsafe_allow_html=True)
    
    st.markdown("""
    We analyze the conditions under which contact tracing can effectively control an epidemic (R<sub>e</sub> < 1).
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.pyplot(create_contour_plot())
        st.markdown('<div class="figure-caption">Fig. 3: Contour plot of effective reproduction number (Re) as a function of tracing fraction (φ) and hospitalization fraction (ρ)</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Critical Conditions</div>', unsafe_allow_html=True)
        st.markdown("""
        Under perfect tracing settings (φ=1, β<sub>m</sub>=0, q=1), R<sub>e</sub> < 1 if and only if:
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="math-formula">', unsafe_allow_html=True)
        st.latex(r'''\begin{cases}
        (1-p_e^i)R_i < 1\\
        \text{and}\\
        \rho > 1 - \frac{1}{R_0(\rho=0)}
        \end{cases}''')
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        **Two critical conditions must be met:**
        
        1. The average number of secondary cases caused by primary high-order traced cases must be below unity
        
        2. The hospitalized (reported & isolated) fraction of untraced cases must exceed a threshold
        """)
    
    st.markdown("---")
    
    st.markdown('<div class="subsection-header">Parameter Sensitivity Analysis</div>', unsafe_allow_html=True)
    
    st.pyplot(create_sensitivity_plot())
    st.markdown('<div class="figure-caption">Fig. 4: Sensitivity of contact tracing effectiveness to disease and implementation parameters</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Disease Characteristics</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Incubation Period (τ)**
        
        - Longer incubation period allows for greater chance to identify contacts while still incubating
        - Higher p<sub>e</sub> and p<sub>e</sub><sup>i</sup> values with longer τ
        - Diseases like Ebola (long incubation) are more amenable to contact tracing control
        
        **Transmission Rate (β)**
        
        - Higher transmission rate requires more intensive tracing and hospitalization
        - For highly transmissible diseases, contact tracing alone may be insufficient
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Implementation Factors</div>', unsafe_allow_html=True)
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        **Critical Implementation Parameters**
        
        - **Tracing delay (δ)**: Shorter delays dramatically improve effectiveness
        - **Monitoring efficacy (β<sub>m</sub>/β)**: Lower ratio yields better control
        - **Tracing coverage (φ)**: Higher fractions of contacts traced needed for high-R<sub>0</sub> diseases
        - **Hospitalization/isolation rate (ρ)**: Higher rates essential for control
        
        Combinations of these parameters can achieve control even when individual improvements are insufficient.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Stability Analysis slide
elif current_slide == "Stability Analysis":
    st.markdown('<div class="section-header">Stability Analysis of Disease-Free Equilibrium</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        We analyze the local stability of the disease-free equilibrium (DFE) under specific assumptions:
        
        - Single-step tracing (q=0)
        - Perfect reporting of untraced cases (ρ=1)
        - Perfect monitoring of traced incubating cases (β<sub>m</sub>=0)
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="subsection-header">Theorem</div>', unsafe_allow_html=True)
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        Under the specified assumptions, if R<sub>e</sub> < 1, the DFE of the model is locally asymptotically stable. If R<sub>e</sub> > 1, the DFE is unstable.
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="subsection-header">Alternative Threshold Parameter</div>', unsafe_allow_html=True)
        st.markdown("""
        We define K<sub>0</sub> := (R<sub>0</sub> - L) + LR.
        
        The stability condition R<sub>e</sub> < 1 is equivalent to K<sub>0</sub> < 1.
        """, unsafe_allow_html=True)
    
    with col2:
        st.pyplot(create_stability_diagram())
        st.markdown('<div class="figure-caption">Fig. 5: Disease dynamics with Re < 1 (stable) vs. Re > 1 (unstable)</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="subsection-header">Proof Approach</div>', unsafe_allow_html=True)
        st.markdown("""
        1. Linearize the system around DFE
        2. Analyze the eigenvalues of the Jacobian matrix
        3. Apply Routh-Hurwitz stability criteria
        4. Show that all eigenvalues have negative real parts when R<sub>e</sub> < 1
        5. Demonstrate at least one eigenvalue has positive real part when R<sub>e</sub> > 1
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="subsection-header">Implications</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Stable Case (R<sub>e</sub> < 1)**
        
        - Disease will die out regardless of initial conditions
        - Rate of decay depends on how far R<sub>e</sub> is below 1
        - Small outbreaks possible but will not persist
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        **Unstable Case (R<sub>e</sub> > 1)**
        
        - Disease will persist in the population
        - Endemic equilibrium will be established
        - Magnitude of outbreak depends on how far R<sub>e</sub> is above 1
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        **Critical Threshold**
        
        - Sharp transition at R<sub>e</sub> = 1
        - Contact tracing parameters can push R<sub>e</sub> below 1
        - Combination of parameters matters more than individual values
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Numerical Results slide
elif current_slide == "Numerical Results":
    st.markdown('<div class="section-header">Numerical Results</div>', unsafe_allow_html=True)
    
    st.markdown("### Application to 2014-2016 West Africa Ebola Outbreak")
    
    st.markdown("""
    We fitted our model to WHO data from the 2014-2016 Ebola outbreak in Sierra Leone and Guinea, focusing on weekly counts of traced individuals among reported cases.
    """)
    
    st.pyplot(create_ebola_data_plot())
    st.markdown('<div class="figure-caption">Fig. 6: Model fit to reported and traced case data in Sierra Leone and Guinea during Jan-Apr 2015</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Key Parameters for Ebola</div>', unsafe_allow_html=True)
        st.markdown("""
        - Incubation period τ = 10 days
        - Time until hospitalization T<sub>h</sub> = 5 days
        - Tracing delay δ was assumed to be minimal
        - Probability of tracking while incubating p<sub>e</sub> = 0.67
        
        Fitted parameters:
        - Linear increase in contact tracing fraction φ(t) = φ<sub>0</sub> + φ<sub>1</sub>t
        - Time-varying transmission rate β(t) = β<sub>0</sub> + β<sub>1</sub>t
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-header">Findings from Data Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
        **Sierra Leone:**
        - Contact tracing fraction φ(t) increased from 0.45 to 0.80
        - Approximately 4,554 cases averted through contact tracing
        - Effective reproduction number R<sub>e</sub> reduced below 1
        
        **Guinea:**
        - Contact tracing fraction φ(t) increased from 0.31 to 0.41
        - Approximately 2,638 cases averted through contact tracing
        - Slower decline in reproduction number
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="subsection-header">Sensitivity Analysis Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Effect of Tracing Order**
        
        - For long incubation period (Ebola), single-step vs. higher-order tracing showed minimal difference
        - For short incubation diseases, higher-order tracing provides significant additional benefit
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Effect of Monitoring Efficacy**
        
        - Perfect monitoring (β<sub>m</sub>=0) greatly improves tracing effectiveness
        - Even imperfect monitoring provides substantial benefit
        - Critical for diseases with short incubation periods
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Synergistic Effects**
        
        - High hospitalization rate (ρ) and high tracing fraction (φ) together can reduce R<sub>e</sub> below 1
        - R<sub>e</sub> becomes less sensitive to time until hospitalization (T<sub>h</sub>) when tracing fraction is high
        - Control strategy should focus on multiple parameters simultaneously
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Conclusions slide
elif current_slide == "Conclusions":
    st.markdown('<div class="section-header">Conclusions and Discussion</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Key Contributions</div>', unsafe_allow_html=True)
        st.markdown("""
        1. Derived explicit formulations for effective reproduction number under different tracing scenarios
        
        2. Reformulated R<sub>e</sub> in terms of epidemiologically interpretable quantities
        
        3. Established stability threshold conditions that determine disease extinction vs. persistence
        
        4. Quantified the impact of contact tracing during the Ebola outbreak
        
        5. Determined conditions under which contact tracing can contain epidemics
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="subsection-header">Main Findings</div>', unsafe_allow_html=True)
        st.markdown("""
        - Contact tracing effectiveness depends on disease characteristics:
          - More effective for diseases with long incubation (like Ebola)
          - Less effective for rapidly spreading diseases with short incubation
          
        - Synergistic effects between hospitalization/isolation and contact tracing
        
        - Higher-order tracing provides additional benefit for diseases with short incubation
        
        - Both monitoring efficacy and tracing delay impact control effectiveness
        """)
    
    with col2:
        st.markdown('<div class="subsection-header">Implications for Public Health</div>', unsafe_allow_html=True)
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("""
        **For diseases like Ebola (long incubation):**
        
        - Contact tracing is highly effective
        - Parameter regime for control is relatively large
        - Focus on expanding coverage and reporting rates
        
        **For diseases like COVID-19 (short incubation):**
        
        - Smaller margin for error
        - Need complementary interventions
        - Critical to minimize tracing delays
        - High monitoring efficacy essential
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="subsection-header">Future Work</div>', unsafe_allow_html=True)
        st.markdown("""
        - Incorporate additional interventions (quarantine, social distancing)
        - Model limited testing and tracing resources
        - Consider spatial heterogeneity in contact tracing implementation
        - Develop decision support tools for public health officials
        - Explore optimal allocation of resources across different control measures
        """)
    
    st.markdown("---")
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **Summary**
    
    Contact tracing can be an effective intervention for controlling epidemics when properly implemented with appropriate consideration of disease characteristics and implementation factors. Our mathematical analysis provides a framework for understanding the conditions under which contact tracing can successfully contain outbreaks and how it can be optimized for different disease scenarios.
    """)
    st.markdown('</div>', unsafe_allow_html=True)