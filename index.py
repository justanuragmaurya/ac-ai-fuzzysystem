import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Fuzzy Logic Cooling Control System",
    page_icon="❄️",
    layout="wide"
)

# App title and description
st.title("Fuzzy Logic Cooling Control System")
st.markdown("""
This application uses fuzzy logic to determine the appropriate cooling level 
based on temperature, humidity, and room occupancy inputs. Adjust the sliders to see how the 
cooling level changes with different environmental conditions.
""")

# Create the fuzzy logic control system
def create_fuzzy_system():
    # Define input variables
    temperature = ctrl.Antecedent(np.arange(15, 35, 1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(30, 80, 1), 'humidity')
    occupancy = ctrl.Antecedent(np.arange(0, 21, 1), 'occupancy')
    cooling = ctrl.Consequent(np.arange(0, 100, 1), 'cooling')

    # Define membership functions for temperature
    temperature['cold'] = fuzz.trimf(temperature.universe, [15, 15, 20])
    temperature['moderate'] = fuzz.trimf(temperature.universe, [18, 25, 28])
    temperature['hot'] = fuzz.trimf(temperature.universe, [25, 35, 35])

    # Define membership functions for humidity
    humidity['low'] = fuzz.trimf(humidity.universe, [30, 30, 50])
    humidity['medium'] = fuzz.trimf(humidity.universe, [40, 60, 60])
    humidity['high'] = fuzz.trimf(humidity.universe, [60, 80, 80])
    
    # Define membership functions for occupancy
    occupancy['low'] = fuzz.trimf(occupancy.universe, [0, 0, 5])
    occupancy['medium'] = fuzz.trimf(occupancy.universe, [3, 8, 12])
    occupancy['high'] = fuzz.trimf(occupancy.universe, [10, 20, 20])

    # Define membership functions for cooling
    cooling['low'] = fuzz.trimf(cooling.universe, [0, 0, 30])
    cooling['medium'] = fuzz.trimf(cooling.universe, [20, 50, 60])
    cooling['high'] = fuzz.trimf(cooling.universe, [50, 100, 100])

    # Define fuzzy rules
    rule1 = ctrl.Rule(temperature['cold'] | humidity['low'], cooling['low'])
    rule2 = ctrl.Rule(temperature['moderate'] & humidity['medium'] & occupancy['low'], cooling['low'])
    rule3 = ctrl.Rule(temperature['moderate'] & humidity['medium'] & occupancy['medium'], cooling['medium'])
    rule4 = ctrl.Rule(temperature['hot'] | humidity['high'], cooling['high'])
    rule5 = ctrl.Rule(occupancy['high'], cooling['high'])

    # Create control system
    cooling_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    return ctrl.ControlSystemSimulation(cooling_ctrl), temperature, humidity, occupancy, cooling

# Create sidebar for inputs
st.sidebar.header("Input Parameters")
temp_value = st.sidebar.slider("Temperature (°C)", 15, 35, 28, 1)
humidity_value = st.sidebar.slider("Humidity (%)", 30, 80, 65, 1)
occupancy_value = st.sidebar.slider("Room Occupancy (people)", 0, 20, 5, 1)

# Create and compute the fuzzy system
cooling_sim, temperature, humidity, occupancy, cooling = create_fuzzy_system()
cooling_sim.input['temperature'] = temp_value
cooling_sim.input['humidity'] = humidity_value
cooling_sim.input['occupancy'] = occupancy_value
cooling_sim.compute()

# Display the result
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Values")
    st.info(f"Temperature: {temp_value} °C")
    st.info(f"Humidity: {humidity_value} %")
    st.info(f"Room Occupancy: {occupancy_value} people")
    
    st.subheader("Output Result")
    cooling_level = cooling_sim.output['cooling']
    st.success(f"Recommended Cooling Level: {cooling_level:.2f}/100")
    
    # Visual indicator
    st.progress(int(cooling_level))
    
    # Cooling recommendation text
    if cooling_level < 30:
        st.write("Recommendation: Low cooling - minimal air conditioning required.")
    elif cooling_level < 60:
        st.write("Recommendation: Medium cooling - moderate air conditioning required.")
    else:
        st.write("Recommendation: High cooling - maximum air conditioning required.")

with col2:
    st.subheader("Fuzzy Logic Visualization")
    
    # Create custom membership function plots rather than using the built-in view() method
    # Temperature membership plot
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    x_temp = np.arange(15, 35, 0.1)
    
    # Plot each membership function
    ax1.plot(x_temp, fuzz.trimf(x_temp, [15, 15, 20]), 'b', linewidth=1.5, label='cold')
    ax1.plot(x_temp, fuzz.trimf(x_temp, [18, 25, 28]), 'g', linewidth=1.5, label='moderate')
    ax1.plot(x_temp, fuzz.trimf(x_temp, [25, 35, 35]), 'r', linewidth=1.5, label='hot')
    
    # Fill current temperature value
    ax1.fill_between([temp_value, temp_value], [0, 1], alpha=0.2)
    ax1.plot([temp_value, temp_value], [0, 1], 'k:', linewidth=1.5)
    
    ax1.set_title('Temperature Membership')
    ax1.legend(loc='center right')
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(15, 35)
    ax1.set_ylabel('Membership')
    ax1.set_xlabel('Temperature (°C)')
    ax1.grid(True)
    st.pyplot(fig1)
    
    # Humidity membership plot
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    x_humid = np.arange(30, 80, 0.1)
    
    # Plot each membership function
    ax2.plot(x_humid, fuzz.trimf(x_humid, [30, 30, 50]), 'b', linewidth=1.5, label='low')
    ax2.plot(x_humid, fuzz.trimf(x_humid, [40, 60, 60]), 'g', linewidth=1.5, label='medium')
    ax2.plot(x_humid, fuzz.trimf(x_humid, [60, 80, 80]), 'r', linewidth=1.5, label='high')
    
    # Fill current humidity value
    ax2.fill_between([humidity_value, humidity_value], [0, 1], alpha=0.2)
    ax2.plot([humidity_value, humidity_value], [0, 1], 'k:', linewidth=1.5)
    
    ax2.set_title('Humidity Membership')
    ax2.legend(loc='center right')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlim(30, 80)
    ax2.set_ylabel('Membership')
    ax2.set_xlabel('Humidity (%)')
    ax2.grid(True)
    st.pyplot(fig2)
    
    # Occupancy membership plot
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    x_occ = np.arange(0, 21, 0.1)
    
    # Plot each membership function
    ax3.plot(x_occ, fuzz.trimf(x_occ, [0, 0, 5]), 'b', linewidth=1.5, label='low')
    ax3.plot(x_occ, fuzz.trimf(x_occ, [3, 8, 12]), 'g', linewidth=1.5, label='medium')
    ax3.plot(x_occ, fuzz.trimf(x_occ, [10, 20, 20]), 'r', linewidth=1.5, label='high')
    
    # Fill current occupancy value
    ax3.fill_between([occupancy_value, occupancy_value], [0, 1], alpha=0.2)
    ax3.plot([occupancy_value, occupancy_value], [0, 1], 'k:', linewidth=1.5)
    
    ax3.set_title('Occupancy Membership')
    ax3.legend(loc='center right')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_xlim(0, 20)
    ax3.set_ylabel('Membership')
    ax3.set_xlabel('Number of People')
    ax3.grid(True)
    st.pyplot(fig3)
    
    # Cooling membership plot
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    x_cool = np.arange(0, 100, 0.1)
    
    # Plot each membership function
    ax4.plot(x_cool, fuzz.trimf(x_cool, [0, 0, 30]), 'b', linewidth=1.5, label='low')
    ax4.plot(x_cool, fuzz.trimf(x_cool, [20, 50, 60]), 'g', linewidth=1.5, label='medium')
    ax4.plot(x_cool, fuzz.trimf(x_cool, [50, 100, 100]), 'r', linewidth=1.5, label='high')
    
    # Fill current output value
    ax4.fill_between([cooling_level, cooling_level], [0, 1], alpha=0.2)
    ax4.plot([cooling_level, cooling_level], [0, 1], 'k:', linewidth=1.5)
    
    ax4.set_title('Cooling Membership')
    ax4.legend(loc='center right')
    ax4.set_ylim(-0.1, 1.1)
    ax4.set_xlim(0, 100)
    ax4.set_ylabel('Membership')
    ax4.set_xlabel('Cooling Level')
    ax4.grid(True)
    st.pyplot(fig4)

# Add explanation section
st.subheader("How It Works")
st.markdown("""
### Fuzzy Logic System
1. **Inputs**: 
   - Temperature (15-35°C)
   - Humidity (30-80%)
   - Room Occupancy (0-20 people)
2. **Output**: Cooling level (0-100)
3. **Rules**:
   - If temperature is cold OR humidity is low, then cooling is low
   - If temperature is moderate AND humidity is medium AND occupancy is low, then cooling is low
   - If temperature is moderate AND humidity is medium AND occupancy is medium, then cooling is medium
   - If temperature is hot OR humidity is high, then cooling is high
   - If occupancy is high, then cooling is high

Fuzzy logic allows us to make decisions based on imprecise inputs, similar to human reasoning.
Each input can partially belong to multiple categories (e.g., a temperature can be both 'moderate' and 'hot' to different degrees).
""")

# Footer
st.markdown("---")
st.caption("Fuzzy Logic Cooling Control System - Powered by scikit-fuzzy and Streamlit")