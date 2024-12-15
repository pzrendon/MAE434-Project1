import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 1.4  # Specific heat ratio for air
R = 287  # Specific gas constant for air (J/kg*K)
T_ambient = 288.15  # Ambient temperature (K)
p_ambient = 101325  # Ambient pressure (Pa)
M_star = 1.0  # Mach number at the throat
M3 = 2.0  # Mach number in the test section
M4 = 0.8  # Mach number at the exit (subsonic after the shock)
D = 0.5  # Tunnel depth (m)

# Design Parameters
A1 = 0.35  # Inlet area (m^2), larger than the throat
A2 = 0.3  # Area before the throat (m^2), slightly larger than the throat
A_star = 0.2  # Throat area (m^2), smallest area
A3 = 0.5 * 0.5  # Test section area (m^2), 0.5m x 0.5m cross-section
A4 = 0.35  # Exit area (m^2)
As = A3 * 0.95  # Area after the normal shock, close to A3

# Functions
def isentropic_temperature(T0, M, gamma):
    return T0 / (1 + ((gamma - 1) / 2) * M**2)

def isentropic_pressure(p0, M, gamma):
    return p0 / (1 + ((gamma - 1) / 2) * M**2) ** (gamma / (gamma - 1))

def isentropic_density(p0, T0, p, T):
    return p / (R * T)

def velocity(M, T):
    return M * np.sqrt(gamma * R * T)

def fan_power(m_dot, T1, T2, eta):
    return m_dot * R * (T2 - T1) / eta

def solve_mach_from_area_ratio(A_ratio, gamma):
    from scipy.optimize import fsolve

    def func(M):
        return (
            1 / M
            * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2))
            ** ((gamma + 1) / (2 * (gamma - 1)))
            - A_ratio
        )

    M_initial_guess = 0.5
    M_solution = fsolve(func, M_initial_guess)
    return M_solution[0]

# Nozzle shape function
def nozzle_shape(x_value, L_converging, L_test_section, L_diverging, A1, A_star, A3, A4):
    if x_value < L_converging:  # Converging section
        return A1 + (A_star - A1) * (x_value / L_converging)
    elif x_value < L_converging + L_test_section:  # Test section
        return A3
    else:  # Diverging section
        return A3 - (A3 - A4) * ((x_value - (L_converging + L_test_section)) / L_diverging)

# Step 1: Stagnation properties
T0 = T_ambient  # Assume isentropic inlet
p0 = p_ambient

# Solve for M1 based on area ratio A1/A*
A_ratio_1_star = A1 / A_star
M1 = solve_mach_from_area_ratio(A_ratio_1_star, gamma)

# Properties at M1
T1 = isentropic_temperature(T0, M1, gamma)
p1 = isentropic_pressure(p0, M1, gamma)
rho1 = isentropic_density(p0, T0, p1, T1)
V1 = velocity(M1, T1)

# Throat properties
T_star = isentropic_temperature(T0, M_star, gamma)
p_star = isentropic_pressure(p0, M_star, gamma)
rho_star = isentropic_density(p0, T0, p_star, T_star)
V_star = velocity(M_star, T_star)

# Test section properties
T3 = isentropic_temperature(T0, M3, gamma)
p3 = isentropic_pressure(p0, M3, gamma)
rho3 = isentropic_density(p0, T0, p3, T3)
V3 = velocity(M3, T3)

# Exit properties
T4 = isentropic_temperature(T0, M4, gamma)
p4 = isentropic_pressure(p0, M4, gamma)
rho4 = isentropic_density(p0, T0, p4, T4)
V4 = velocity(M4, T4)

# Step 2: Mass flow rate
mass_flow_rate = rho_star * A_star * V_star

# Fan properties
eta_fan = 0.92  # Adiabatic efficiency
p02 = p_star  # Assume stagnation pressure is constant across the fan
T02 = T0 * (1 + ((gamma - 1) / 2) * M_star**2)  # Total temperature after fan
fan_power_required = fan_power(mass_flow_rate, T0, T02, eta_fan)

# Step 3: Length definitions
x_total = 4.0  # Total length of the wind tunnel (m)
L_converging = 1.5  # Length of converging section (m)
L_test_section = 1.0  # Test section length (m)
L_diverging = 1.5  # Length of diverging section (m)

# Step 8: Force on bolts and foundation
F_thrust = mass_flow_rate * V4 + (p4 - p_ambient) * A4

# Cross-sectional area distribution
x = np.linspace(0, x_total, 500)
y_smooth = np.piecewise(
    x,
    [x < L_converging, (x >= L_converging) & (x < L_converging + L_test_section), x >= L_converging + L_test_section],
    [
        lambda x: A1 + (A_star - A1) * (x / L_converging),
        lambda x: A3,
        lambda x: A3 - (A3 - A4) * ((x - (L_converging + L_test_section)) / L_diverging),
    ],
)

# Generate 10 discrete x,y data points
x_discrete = np.linspace(0, x_total, 10)
y_discrete = [nozzle_shape(xi, L_converging, L_test_section, L_diverging, A1, A_star, A3, A4) for xi in x_discrete]

# Piecewise properties for graphs
Mach_numbers = np.piecewise(x, [x < L_converging, (x >= L_converging) & (x < L_converging + L_test_section), x >= L_converging + L_test_section], [
    lambda xi: M1 + (M_star - M1) * (xi / L_converging),  # Transition from M1 to M_star
    M3,
    M4
])
temperatures = np.piecewise(x, [x < L_converging, (x >= L_converging) & (x < L_converging + L_test_section), x >= L_converging + L_test_section], [
    lambda xi: isentropic_temperature(T0, M1 + (M_star - M1) * (xi / L_converging), gamma),
    T3,
    T4
])
pressures = np.piecewise(x, [x < L_converging, (x >= L_converging) & (x < L_converging + L_test_section), x >= L_converging + L_test_section], [
    lambda xi: isentropic_pressure(p0, M1 + (M_star - M1) * (xi / L_converging), gamma),
    p3,
    p4
])
densities = np.piecewise(x, [x < L_converging, (x >= L_converging) & (x < L_converging + L_test_section), x >= L_converging + L_test_section], [
    lambda xi: isentropic_density(p0, T0, 
                                  isentropic_pressure(p0, M1 + (M_star - M1) * (xi / L_converging), gamma),
                                  isentropic_temperature(T0, M1 + (M_star - M1) * (xi / L_converging), gamma)),
    rho3,
    rho4
])
velocities = np.piecewise(x, [x < L_converging, (x >= L_converging) & (x < L_converging + L_test_section), x >= L_converging + L_test_section], [
    lambda xi: velocity(M1 + (M_star - M1) * (xi / L_converging),
                        isentropic_temperature(T0, M1 + (M_star - M1) * (xi / L_converging), gamma)),
    V3,
    V4
])

# Conservation laws
mass_flow_inlet = rho_star * A_star * V_star
mass_flow_exit = rho4 * A4 * V4
energy_conservation_inlet = 0.5 * rho_star * V_star**2 + p_star / rho_star
energy_conservation_exit = 0.5 * rho4 * V4**2 + p4 / rho4

# Print results
print("Cross-sectional Areas:")
print(f"Inlet Area (A1): {A1:.2f} m^2")
print(f"Pre-Throat Area (A2): {A2:.2f} m^2")
print(f"Throat Area (A*): {A_star:.2f} m^2")
print(f"Test Section Area (A3): {A3:.2f} m^2")
print(f"Post-Shock Area (As): {As:.2f} m^2")
print(f"Exit Area (A4): {A4:.2f} m^2")

print("\nLengths:")
print(f"Converging Section Length: {L_converging:.2f} m")
print(f"Diverging Section Length: {L_diverging:.2f} m")
print(f"Test Section Length: {L_test_section:.2f} m")

print("\nFan Properties:")
print(f"Fan Total Pressure Ratio (p02/p01): {p02/p0:.2f}")
print(f"Fan Power Required: {fan_power_required / 1000:.2f} kW")

print("\nReynolds Number:")
mu = 1.81e-5  # Dynamic viscosity of air (Pa.s)
Re_test_section = (rho3 * V3 * D) / mu
print(f"Reynolds Number in Test Section: {Re_test_section:.2e}")

print("\nProperties at M1:")
print(f"Mach Number at M1: {M1:.2f}")
print(f"Temperature at M1: {T1:.2f} K")
print(f"Pressure at M1: {p1:.2f} Pa")
print(f"Density at M1: {rho1:.2f} kg/m^3")
print(f"Velocity at M1: {V1:.2f} m/s")

print("\nConservation Laws:")
print(f"Mass Flow Rate at Inlet: {mass_flow_inlet:.2f} kg/s")
print(f"Mass Flow Rate at Exit: {mass_flow_exit:.2f} kg/s")
print(f"Energy Conservation at Inlet: {energy_conservation_inlet:.2f} J")
print(f"Energy Conservation at Exit: {energy_conservation_exit:.2f} J")

print("\nForce on Bolts and Foundation")
print(f"Force due to thrust: {F_thrust:.2f} N")

# Generate nozzle shape plot
plt.figure(figsize=(10, 6))
plt.plot(x, y_smooth, label="Upper Profile")
plt.plot(x, -y_smooth, label="Lower Profile")
plt.fill_between(x, -y_smooth, y_smooth, alpha=0.2, label="Nozzle Area")
plt.axvline(L_converging, color='green', linestyle='--', label="Throat (A*)")
plt.axvline(L_converging + L_test_section, color='red', linestyle='--', label="End Test Section")
plt.title("Wind Tunnel Schematic")
plt.xlabel("Length (m)")
plt.ylabel("Height (m)")
plt.legend()
plt.grid()
plt.show()

# Generate plots for flow properties
plt.figure(figsize=(10, 12))

plt.subplot(5, 1, 1)
plt.plot(x, Mach_numbers, label="Mach Number")
plt.axvline(L_converging, color='green', linestyle='--', label="Throat (A*)")
plt.axvline(L_converging + L_test_section, color='red', linestyle='--', label="End Test Section")
plt.ylabel("Mach Number")
plt.title("Mach Changes Along Wind Tunnel Center-Line")
plt.grid()
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(x, pressures, label="Pressure")
plt.axvline(L_converging, color='green', linestyle='--', label="Throat (A*)")
plt.axvline(L_converging + L_test_section, color='red', linestyle='--', label="End Test Section")
plt.ylabel("Pressure (Pa)")
plt.title("Pressure Changes Along Wind Tunnel Center-Line")
plt.grid()
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(x, densities, label="Density")
plt.axvline(L_converging, color='green', linestyle='--', label="Throat (A*)")
plt.axvline(L_converging + L_test_section, color='red', linestyle='--', label="End Test Section")
plt.ylabel("Density (kg/m^3)")
plt.title("Density Changes Along Wind Tunnel Center-Line")
plt.grid()
plt.legend()

plt.subplot(5, 1, 4)
plt.plot(x, temperatures, label="Temperature")
plt.axvline(L_converging, color='green', linestyle='--', label="Throat (A*)")
plt.axvline(L_converging + L_test_section, color='red', linestyle='--', label="End Test Section")
plt.ylabel("Temperature (K)")
plt.title("Temperature Changes Along Wind Tunnel Center-Line")
plt.grid()
plt.legend()

plt.subplot(5, 1, 5)
plt.plot(x, velocities, label="Velocity")
plt.axvline(L_converging, color='green', linestyle='--', label="Throat (A*)")
plt.axvline(L_converging + L_test_section, color='red', linestyle='--', label="End Test Section")
plt.xlabel("Length (m)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Changes Along Wind Tunnel Center-Line")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
