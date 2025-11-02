# missioncalc_ai.py
# 🚀 MissionCalc AI — NASA-Level Propulsion Calculator Dashboard
# Runs entirely offline with Streamlit

import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import pandas as pd
from scipy import integrate
import io

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="MissionCalc AI", layout="wide", page_icon="🚀")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E3A8A;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 MissionCalc AI</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Spaceflight Calculator — NASA-Grade Propulsion Analysis")

st.markdown("""
This tool performs professional-grade rocket-engine and mission calculations.
Enter your data below and let MissionCalc do the rest.
""")

# -------------------- SIDEBAR FOR SETTINGS --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    analysis_mode = st.selectbox(
        "Analysis Mode",
        ["Standard Mission", "Engine Performance", "Trajectory Analysis", "Custom"]
    )
    
    g0 = st.number_input("Gravity Constant (m/s²)", value=9.80665, step=0.01)
    show_advanced = st.checkbox("Show Advanced Options")
    
    if show_advanced:
        st.subheader("Advanced Parameters")
        expansion_ratio = st.number_input("Nozzle Expansion Ratio", value=40.0)
        chamber_temp = st.number_input("Chamber Temperature (K)", value=3500.0)
        gamma = st.number_input("Specific Heat Ratio (γ)", value=1.2)

# -------------------- USER INPUT --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚀 Propulsion Data")
    thrust_values = st.text_input("Enter Thrust values (kN, comma separated)", "760, 780, 770, 755, 765")
    isp = st.number_input("Specific Impulse (s)", value=310.0, min_value=0.0)
    chamber_pressure = st.number_input("Chamber Pressure (MPa)", value=7.5)
    
with col2:
    st.subheader("📊 Mass Properties")
    prop_mass = st.number_input("Propellant Mass (kg)", value=250000.0, min_value=0.0)
    dry_mass = st.number_input("Dry Mass (kg)", value=90000.0, min_value=0.0)
    payload_mass = st.number_input("Payload Mass (kg)", value=10000.0, min_value=0.0)
    
with col3:
    st.subheader("⏱️ Mission Parameters")
    burn_time = st.number_input("Burn Time (s)", value=325.0, min_value=0.0)
    p_values = st.text_input("Enter Chamber Pressure values (MPa, comma separated)", "7.5, 7.6, 7.4, 7.5")
    mission_type = st.selectbox(
        "Mission Type",
        ["LEO Launch", "GTO Injection", "Lunar Transfer", "Mars Transfer", "Custom"]
    )

st.markdown("---")
calc_btn = st.button("🧮 Compute Mission Analysis", type="primary", use_container_width=True)

# -------------------- COMPUTATION FUNCTIONS --------------------
def calculate_rocket_performance(thrust_list, pressure_list, prop_mass, dry_mass, 
                               payload_mass, burn_time, isp, g0):
    """Calculate comprehensive rocket performance parameters"""
    
    # Convert inputs
    thrust_array = np.array([float(x) for x in thrust_list]) * 1000  # kN to N
    pressure_array = np.array([float(x) for x in pressure_list]) * 1e6  # MPa to Pa
    
    # Basic calculations
    avg_thrust = np.mean(thrust_array)
    avg_pressure = np.mean(pressure_array)
    
    # Mass calculations
    m0 = dry_mass + prop_mass + payload_mass  # Initial mass
    mf = dry_mass + payload_mass  # Final mass
    mass_ratio = m0 / mf
    
    # Performance calculations
    delta_v = isp * g0 * math.log(m0 / mf)
    mdot = prop_mass / burn_time
    calc_isp = avg_thrust / (mdot * g0)
    tw_ratio = avg_thrust / (m0 * g0)
    
    # Additional parameters
    total_impulse = avg_thrust * burn_time
    mass_flow_rate = mdot
    exhaust_velocity = isp * g0
    
    # Thrust variation analysis
    thrust_std = np.std(thrust_array)
    thrust_cv = (thrust_std / avg_thrust) * 100  # Coefficient of variation
    
    return {
        'avg_thrust': avg_thrust,
        'avg_pressure': avg_pressure,
        'mass_ratio': mass_ratio,
        'delta_v': delta_v,
        'mdot': mdot,
        'calc_isp': calc_isp,
        'tw_ratio': tw_ratio,
        'total_impulse': total_impulse,
        'mass_flow_rate': mass_flow_rate,
        'exhaust_velocity': exhaust_velocity,
        'thrust_std': thrust_std,
        'thrust_cv': thrust_cv,
        'm0': m0,
        'mf': mf
    }

def create_thrust_profile_plot(thrust_values, burn_time):
    """Create detailed thrust profile visualization"""
    thrust_array = np.array([float(x) for x in thrust_values])
    time_steps = np.linspace(0, burn_time, len(thrust_array))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Thrust profile
    ax1.plot(time_steps, thrust_array, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.fill_between(time_steps, thrust_array, alpha=0.3, color='blue')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Thrust (kN)')
    ax1.set_title('Thrust Profile Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Thrust histogram
    ax2.hist(thrust_array, bins=10, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(thrust_array), color='red', linestyle='--', 
                label=f'Mean: {np.mean(thrust_array):.1f} kN')
    ax2.set_xlabel('Thrust (kN)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Thrust Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_mass_pie_chart(dry_mass, prop_mass, payload_mass):
    """Create mass distribution pie chart"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    masses = [dry_mass, prop_mass, payload_mass]
    labels = ['Dry Mass', 'Propellant Mass', 'Payload Mass']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    wedges, texts, autotexts = ax.pie(masses, labels=labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    
    # Improve autotext styling
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Mass Distribution')
    plt.tight_layout()
    return fig

def generate_mission_insights(results, mission_type):
    """Generate AI-powered mission insights"""
    insights = []
    
    # T/W ratio analysis
    if results['tw_ratio'] < 1:
        insights.append(("danger", "🚨 CRITICAL: Thrust-to-Weight Ratio < 1 — Vehicle cannot lift off!"))
    elif results['tw_ratio'] < 1.2:
        insights.append(("warning", "⚠️ MARGINAL: T/W ratio is low — Consider reducing mass or increasing thrust"))
    elif results['tw_ratio'] < 1.5:
        insights.append(("success", "✅ ADEQUATE: T/W ratio suitable for upper-stage operations"))
    else:
        insights.append(("success", "🎯 EXCELLENT: Strong T/W ratio — Ideal for launch phase"))
    
    # Delta-V analysis
    delta_v_requirements = {
        "LEO Launch": 9400,
        "GTO Injection": 2400,
        "Lunar Transfer": 3200,
        "Mars Transfer": 3600
    }
    
    if mission_type in delta_v_requirements:
        req_dv = delta_v_requirements[mission_type]
        if results['delta_v'] < req_dv:
            insights.append(("danger", f"❌ INSUFFICIENT ΔV: {results['delta_v']:.0f} m/s vs required {req_dv} m/s for {mission_type}"))
        else:
            insights.append(("success", f"✅ SUFFICIENT ΔV: {results['delta_v']:.0f} m/s meets {mission_type} requirements"))
    
    # Thrust stability analysis
    if results['thrust_cv'] > 10:
        insights.append(("warning", f"⚠️ HIGH THRUST VARIATION: {results['thrust_cv']:.1f}% CV indicates potential combustion instability"))
    else:
        insights.append(("success", f"✅ STABLE THRUST: {results['thrust_cv']:.1f}% variation within acceptable limits"))
    
    # Mass ratio analysis
    if results['mass_ratio'] > 10:
        insights.append(("success", f"✅ EXCELLENT MASS RATIO: {results['mass_ratio']:.2f} indicates efficient design"))
    elif results['mass_ratio'] > 5:
        insights.append(("success", f"✅ GOOD MASS RATIO: {results['mass_ratio']:.2f} provides adequate performance"))
    else:
        insights.append(("warning", f"⚠️ LOW MASS RATIO: {results['mass_ratio']:.2f} may limit mission capability"))
    
    return insights

# -------------------- MAIN COMPUTATION --------------------
if calc_btn:
    try:
        # Perform calculations
        results = calculate_rocket_performance(
            thrust_values.split(","), 
            p_values.split(","), 
            prop_mass, dry_mass, payload_mass, 
            burn_time, isp, g0
        )
        
        # -------------------- RESULTS DISPLAY --------------------
        st.subheader("📊 Mission Analysis Results")
        
        # Key Metrics in columns
        colA, colB, colC, colD = st.columns(4)
        
        with colA:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Average Thrust", f"{results['avg_thrust']/1000:.1f} kN")
            st.metric("Total Impulse", f"{results['total_impulse']/1e6:.0f} MN·s")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with colB:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("ΔV (Delta-V)", f"{results['delta_v']:.0f} m/s")
            st.metric("Mass Ratio", f"{results['mass_ratio']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with colC:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Thrust/Weight Ratio", f"{results['tw_ratio']:.3f}")
            st.metric("Mass Flow Rate", f"{results['mass_flow_rate']:.1f} kg/s")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with colD:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Specific Impulse", f"{results['calc_isp']:.1f} s")
            st.metric("Exhaust Velocity", f"{results['exhaust_velocity']/1000:.1f} km/s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # -------------------- VISUALIZATIONS --------------------
        st.subheader("📈 Performance Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            thrust_fig = create_thrust_profile_plot(thrust_values.split(","), burn_time)
            st.pyplot(thrust_fig)
            
        with viz_col2:
            mass_fig = create_mass_pie_chart(dry_mass, prop_mass, payload_mass)
            st.pyplot(mass_fig)
        
        # -------------------- AI INSIGHTS --------------------
        st.subheader("🧠 AI Mission Analysis")
        
        insights = generate_mission_insights(results, mission_type)
        
        for insight_type, message in insights:
            if insight_type == "success":
                st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
            elif insight_type == "warning":
                st.markdown(f'<div class="warning-box">{message}</div>', unsafe_allow_html=True)
            elif insight_type == "danger":
                st.markdown(f'<div class="danger-box">{message}</div>', unsafe_allow_html=True)
        
        # -------------------- DATA EXPORT --------------------
        st.subheader("💾 Export Results")
        
        # Create DataFrame for export
        export_data = {
            'Parameter': [
                'Mission Type', 'Average Thrust (kN)', 'Average Pressure (MPa)',
                'Delta-V (m/s)', 'Thrust/Weight Ratio', 'Specific Impulse (s)',
                'Mass Ratio', 'Total Impulse (MN·s)', 'Mass Flow Rate (kg/s)',
                'Initial Mass (kg)', 'Final Mass (kg)', 'Burn Time (s)'
            ],
            'Value': [
                mission_type, results['avg_thrust']/1000, results['avg_pressure']/1e6,
                results['delta_v'], results['tw_ratio'], results['calc_isp'],
                results['mass_ratio'], results['total_impulse']/1e6, results['mass_flow_rate'],
                results['m0'], results['mf'], burn_time
            ]
        }
        
        df_export = pd.DataFrame(export_data)
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            # CSV Download
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="missioncalc_analysis.csv",
                mime="text/csv",
            )
        
        with col_exp2:
            # Show data table
            if st.checkbox("Show Data Table"):
                st.dataframe(df_export, use_container_width=True)
                
    except Exception as e:
        st.error(f"🚨 Computation Error: {str(e)}")
        st.info("💡 Please check your input data format. Make sure all numeric values are properly formatted.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🚀 MissionCalc AI — Professional Spaceflight Analysis Tool<br>
    For educational and research purposes • Not certified for actual flight operations
    </div>
    """, 
    unsafe_allow_html=True
)