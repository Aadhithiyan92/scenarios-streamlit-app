# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# Load placeholder data (replace with real data)
data = pd.DataFrame({
    'State': ['California', 'Texas', 'New York', 'Florida', 'Colorado'],
    'BEV Adoption (%)': [12.5, 3.2, 5.4, 2.9, 6.8],
    'CO2 Reduction (tons)': [32000, 8000, 15000, 7000, 10000]
})

st.set_page_config(page_title="Nationwide ZEV Dashboard", layout="wide")

st.title("🚗 Nationwide ZEV Analysis Dashboard")

st.markdown("""
Welcome to the Zero-Emission Vehicle (ZEV) adoption dashboard. This tool allows you to:
- Compare adoption trends across states
- Evaluate CO₂ emissions reductions
- Explore policy scenario simulations

_This is a demonstration layout. Real-time data and interactive modeling results will be integrated in the next version._
""")

st.subheader("🔍 State-wise BEV Adoption")
fig1 = px.bar(data, x='State', y='BEV Adoption (%)', color='State', title="Battery Electric Vehicle Adoption by State")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🌿 CO₂ Emissions Reductions")
fig2 = px.pie(data, names='State', values='CO2 Reduction (tons)', title="CO₂ Reductions by State")
st.plotly_chart(fig2, use_container_width=True)

st.sidebar.header("🧪 Scenario Filters")
st.sidebar.markdown("(Simulation options coming soon)")
st.sidebar.selectbox("Choose a Policy Focus", ["Growth Investment", "Incentive Support", "Infrastructure Expansion"])
st.sidebar.slider("Time Horizon (years)", 2025, 2035, 2030)

st.markdown("---")
st.caption("© 2025 Nationwide ZEV Initiative | Built with Streamlit")
