# MAE434 Project 1 Rev-7
import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 1.4  # Specific heat ratio for air
R = 287  # Specific gas constant for air (J/kg*K)
T_a = 288.15  # Ambient temperature (K)
p_a = 101325  # Ambient pressure (Pa)
A3 = 1.0 * 1.0  # Test section area (m^2)
M3 = 0.6  # Design Mach number in the test section
M4 = 0.2 # Assumed Mach number at exit
V3 = M3 * 340.3 # Veloity of Mach number at SL
A1 = A2 = 1.5  # Cross-sectional areas (m^2)
A4 = 2  # Area continuity
L3 = 3.0  # Length of test section (m)
rho_a = p_a / (R * T_a)  # Ambient density (kg/m³)

# Functions
def isentropic_temperature(T0, M, gamma):
    return T0 / (1 + ((gamma - 1) / 2) * M**2)

# Step 1: Calculate cross-sectional areas
areas = {"A1": A1, "A2": A2, "A3": A3, "A4": A4}

# Step 2: Calculate lengths of converging and diverging sections
L_converging = 2  # Assumed length for converging section (m)
L_diverging = 3  # Assumed length for diverging section (m)

# Step 3: Calculate the total pressure ratio (p02/p01) and fan power
p03 = p_a * (1 + (gamma - 1) / 2 * M3**2) ** (gamma / (gamma - 1))  # Total pressure at test section
T03 = T_a * (1 + (gamma - 1) / 2 * M3**2)  # Total temperature at test section

# Assume fan provides total pressure increase
p02 = p03  # Total pressure at the exit of the fan
T02 = T03  # Total temperature at the exit of the fan
p_ratio = p02 / p_a  # Pressure ratio across the fan

# Mass flow rate
V3 = M3 * np.sqrt(gamma * R * T_a)
mass_flow_rate_test = rho_a * A3 * V3

# Fan power calculation
# mass_flow_rate = rho * A3 * V3  #(p03 * A3) / (R * T03) * np.sqrt(gamma / R) * M3
power = mass_flow_rate_test * R * (T02 - T_a)

# Step 4: Define building size and plot schematic
building_length = L_converging + L3 + L_diverging
building_width = max(A1**0.5, A4**0.5) + 1  # Add extra space for infrastructure

# Schematic plot
x_schematic = [0, L_converging, L_converging + L3, building_length]
y_schematic = [A1, A3, A3, A4]
plt.figure(figsize=(10, 6))
plt.plot(x_schematic, y_schematic, label="Wind Tunnel Profile")
plt.fill_between(x_schematic, 0, y_schematic, alpha=0.2, label="Tunnel Area")
plt.title("Wind Tunnel Schematic (Half View)")
plt.xlabel("Length (m)")
plt.ylabel("Cross-sectional Area (m²)")
plt.legend()
plt.grid()
plt.show()

# Double-sided schematic plot
plt.figure(figsize=(10, 6))
plt.plot(x_schematic, y_schematic, label="Upper Profile")
plt.plot(x_schematic, [-y for y in y_schematic], label="Lower Profile", linestyle="-", color='green')
plt.fill_between(x_schematic, [-y for y in y_schematic], y_schematic, alpha=0.2, label="Tunnel Area")
plt.title("Wind Tunnel Schematic (Full View)")
plt.xlabel("Length (m)")
plt.ylabel("Height (m)")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add centerline
plt.legend()
plt.grid()
plt.show()

# Step 5: Reynolds number calculation
rho3 = p03 / (R * T03)  # Density in the test section
V3 = M3 * np.sqrt(gamma * R * T03)  # Velocity in the test section
Re = rho3 * V3 * 1.0 / (1.81e-5)  # Assume dynamic viscosity = 1.81e-5 Pa·s

# Step 6: Verify mass and energy conservation
mass_conservation = mass_flow_rate_test
energy_conservation1 = (T02 - T_a) * mass_flow_rate_test * R
energy_conservation2 = (T03 - T_a) * mass_flow_rate_test * R

# Temperatures
T01 = T_a  # Total temperature at inlet (K)
T03 = isentropic_temperature(T01, M3, gamma)
T04 = isentropic_temperature(T01, M4, gamma)

# Smooth transition for the nozzle shape using a sinusoidal function
# Re-define `x` as it was likely lost during the environment reset
x = np.linspace(0, building_length, 500)

# Smooth transition for the nozzle shape using a sinusoidal function
def smooth_transition(x, x_start, x_end, A_start, A_end):
    """Generate smooth rounded transitions between two points."""
    return A_start + (A_end - A_start) * 0.5 * (1 - np.cos(np.pi * (x - x_start) / (x_end - x_start)))

y_smooth = np.piecewise(x, 
                        [x < L_converging, 
                         (x >= L_converging) & (x < L_converging + L3), 
                         x >= L_converging + L3],
                        [lambda x: smooth_transition(x, 0, L_converging, A1, A3),
                         lambda x: A3,
                         lambda x: smooth_transition(x, L_converging + L3, building_length, A3, A4)])

# Double-sided smooth nozzle shape with rounded transitions
plt.figure(figsize=(10, 6))
plt.plot(x, y_smooth, label="Upper Profile (Smooth)")
plt.plot(x, -y_smooth, label="Lower Profile (Smooth)", linestyle="-", color='green')
plt.fill_between(x, -y_smooth, y_smooth, alpha=0.2, label="Nozzle Area")
plt.title("Nozzle Shape Function y(x) (Smooth Rounded Transitions)")
plt.xlabel("Length (m)")
plt.ylabel("Height (m)")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add centerline
plt.legend()
plt.grid()
plt.show()



# Step 8: Quasi-1D flow calculations along the centerline
M = np.linspace(0.2, 0.8, 500)
p = p_a * (1 + (gamma - 1) / 2 * M**2) ** (-gamma / (gamma - 1))
T = T_a * (1 + (gamma - 1) / 2 * M**2) ** -1
rho = p / (R * T)
V = M * np.sqrt(gamma * R * T)

# Plot M, p, T, rho, and V as separate graphs on one page
fig, axs = plt.subplots(5, 1, figsize=(10, 20))

axs[0].plot(x, M, label="Mach number (M)")
axs[0].set_title("Mach Number (M) Along Centerline")
axs[0].set_xlabel("Length (m)")
axs[0].set_ylabel("Mach number")
axs[0].grid()

axs[1].plot(x, p, label="Pressure (p)", color='orange')
axs[1].set_title("Pressure (p) Along Centerline")
axs[1].set_xlabel("Length (m)")
axs[1].set_ylabel("Pressure (Pa)")
axs[1].grid()

axs[2].plot(x, T, label="Temperature (T)", color='green')
axs[2].set_title("Temperature (T) Along Centerline")
axs[2].set_xlabel("Length (m)")
axs[2].set_ylabel("Temperature (K)")
axs[2].grid()

axs[3].plot(x, rho, label="Density (ρ)", color='red')
axs[3].set_title("Density (ρ) Along Centerline")
axs[3].set_xlabel("Length (m)")
axs[3].set_ylabel("Density (kg/m³)")
axs[3].grid()

axs[4].plot(x, V, label="Velocity (V)", color='purple')
axs[4].set_title("Velocity (V) Along Centerline")
axs[4].set_xlabel("Length (m)")
axs[4].set_ylabel("Velocity (m/s)")
axs[4].grid()

plt.tight_layout()
plt.show()

# Print outputs for steps 1-9
# Step 1: Cross-sectional areas
print("Step 1: Cross-sectional Areas")
for section, area in areas.items():
    print(f"{section} = {area:.2f} m²")

# Step 2: Lengths of converging and diverging sections
print("\nStep 2: Section Lengths")
print(f"Converging section length = {L_converging:.2f} m")
print(f"Diverging section length = {L_diverging:.2f} m")
print(f"Test section length = {L3:.2f} m")

# Step 3: Total pressure ratio and fan power
print("\nStep 3: Fan Properties")
print(f"Total pressure ratio (p02/p01) = {p_ratio:.2f}")
print(f"Fan power = {power / 1000:.2f} kW")

# Step 4: Building size
print("\nStep 4: Building Dimensions")
print(f"Building length = {building_length:.2f} m")
print(f"Building width = {building_width:.2f} m")

# Step 5: Reynolds number
print("\nStep 5: Reynolds Number")
print(f"Reynolds number in the test section = {Re:.2e}")

# Step 6: Mass and energy conservation
print("\nStep 6: Conservation Laws")
print(f"Mass flow rate = {mass_conservation:.2f} kg/s")
print(f"Energy at Inlet = {energy_conservation1 / 1000:.2f} kJ")
print(f"Energy at Outlet = {energy_conservation2 / 1000:.2f} kJ")
print(f"Temperatures: T03 = {T03:.2f} K, T04 = {T04:.2f} K")

# Step 7: Smooth half-height function
print("\nStep 7: Smooth Nozzle Shape")
print("Smooth half-height function y(x) defined and plotted.")

# Step 8: Quasi-1D flow properties
print("\nStep 8: Quasi-1D Flow Properties")
print("M, p, T, ρ, and V plotted along the centerline.")

# Step 9: Bolt force
thrust = (p03 - p_a) * A3  # Thrust from pressure difference
print("\nStep 9: Bolt Force Calculation")
print(f"Force on bolts and foundation = {thrust:.2f} N")


# Extra Credit (Simplied CFD Model)
# Constants and parameters
p_a = 101325  # Ambient pressure (Pa)
rho_a = 1.225  # Ambient density (kg/m³)
A = np.piecewise(x, 
                 [x < L_converging, 
                  (x >= L_converging) & (x < L_converging + L3), 
                  x >= L_converging + L3],
                 [lambda x: smooth_transition(x, 0, L_converging, A1, A3),
                  lambda x: A3,
                  lambda x: smooth_transition(x, L_converging + L3, building_length, A3, A4)])

# Velocity and pressure distribution based on continuity and Bernoulli's equation
V_a = 50  # Arbitrary initial velocity at the inlet (m/s)
m = V_a * A1  # Mass flow rate (constant)
V = m / A  # Velocity distribution

# Pressure distribution using Bernoulli's principle
p = p_a + 0.5 * rho_a * (V_a**2 - V**2)

# Plot velocity distribution
plt.figure(figsize=(10, 6))
plt.plot(x, V, label="Velocity (V)", color='blue')
plt.title("Velocity Distribution Along the Wind Tunnel")
plt.xlabel("Length (m)")
plt.ylabel("Velocity (m/s)")
plt.grid()
plt.legend()
plt.show()

# Plot pressure distribution
plt.figure(figsize=(10, 6))
plt.plot(x, p, label="Pressure (p)", color='green')
plt.title("Pressure Distribution Along the Wind Tunnel")
plt.xlabel("Length (m)")
plt.ylabel("Pressure (Pa)")
plt.grid()
plt.legend()
plt.show()

