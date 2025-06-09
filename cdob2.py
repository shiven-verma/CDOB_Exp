"""
CDOB (Complementary Disturbance Observer Based) Control System
Double Integrator with Output Delay and PD Control
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System parameters
alpha = 5.0                   # PD controller parameter
Kp, Kd = alpha**2, 2*alpha    # PD gains
tau_d = 0.1                   # Output delay (seconds)
omega_net = 50.0              # CDOB filter cutoff frequency (rad/s)

# Simulation parameters
t_end, dt = 5.0, 0.001
t = np.arange(0, t_end, dt)
n_steps = len(t)
r = np.ones_like(t)           # Unit step reference

class DoubleIntegratorSystem:
    def __init__(self, dt, tau_d=0):
        self.dt = dt
        self.tau_d = tau_d
        self.delay_buffer_size = int(tau_d / dt) if tau_d > 0 else 0
        self.reset()
    
    def reset(self):
        self.x = 0.0  # Position
        self.v = 0.0  # Velocity
        self.state = np.array([0.0, 0.0])  # [position, velocity]
        # Delay buffer for output
        self.delay_buffer = [0.0] * max(1, self.delay_buffer_size)
        
    def dynamics(self, t, state, u):
        """System dynamics for solve_ivp: [x_dot, v_dot] = [v, u]"""
        x, v = state
        return [v, u]
        
    def step(self, u):
        # Use solve_ivp for more robust integration
        t_span = [0, self.dt]
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, u),
            t_span,
            self.state,
            method='RK45',  # Runge-Kutta 4(5) method
            rtol=1e-8,
            atol=1e-10
        )
        
        # Update state with the final solution
        self.state = sol.y[:, -1]
        self.x, self.v = self.state
        
        # Apply output delay
        if self.delay_buffer_size > 0:
            self.delay_buffer.append(self.x)
            x_delayed = self.delay_buffer.pop(0)
        else:
            x_delayed = self.x
            
        return self.x, self.v, x_delayed

# PD Controller
class PDController:
    def __init__(self, Kp, Kd, dt):
        self.Kp = Kp
        self.Kd = Kd
        self.dt = dt
        self.prev_error = 0.0
        
    def control(self, reference, measurement):
        error = reference - measurement
        error_dot = (error - self.prev_error) / self.dt
        self.prev_error = error
        
        u = self.Kp * error + self.Kd * error_dot
        return u

# Low Pass Filter
class LowPassFilter:
    def __init__(self, omega_c, dt):
        self.omega_c = omega_c
        self.dt = dt
        self.y_prev = 0.0
        
    def filter(self, x):
        # If omega_c is zero, no frequency passes through (perfect filter)
        if self.omega_c == 0.0:
            return 0.0
        
        # First-order low pass filter: y_dot = -omega_c * y + omega_c * x
        # Discretized using Euler method
        y = self.y_prev + self.omega_c * (x - self.y_prev) * self.dt
        self.y_prev = y
        return y

# Numerical differentiation class
class Differentiator:
    def __init__(self, dt):
        self.dt = dt
        self.history = []
        
    def differentiate(self, x):
        self.history.append(x)
        if len(self.history) < 3:
            return 0.0
        elif len(self.history) > 5:
            self.history.pop(0)
        
        # Second derivative using finite differences
        if len(self.history) >= 3:
            # Using 3-point finite difference formula
            x_ddot = (self.history[-1] - 2*self.history[-2] + self.history[-3]) / (self.dt**2)
            return x_ddot
        return 0.0

# Consistent integrator using same RK45 method as the plant
class ConsistentIntegrator:
    def __init__(self, dt):
        self.dt = dt
        self.state = np.array([0.0, 0.0])  # [velocity, position]
        
    def dynamics(self, t, state, acceleration):
        """Integration dynamics: [v_dot, p_dot] = [acceleration, velocity]"""
        velocity, position = state
        return [acceleration, velocity]
        
    def double_integrate(self, acceleration):
        """Double integrate using RK45 method (same as plant)"""
        t_span = [0, self.dt]
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, acceleration),
            t_span,
            self.state,
            method='RK45',  # Same method as plant
            rtol=1e-8,      # Same tolerances as plant
            atol=1e-10
        )
        
        # Update state with the final solution
        self.state = sol.y[:, -1]
        velocity, position = self.state
        return position
        
    def reset(self):
        self.state = np.array([0.0, 0.0])

# Initialize systems
system_no_delay = DoubleIntegratorSystem(dt, tau_d=0)
system_with_delay = DoubleIntegratorSystem(dt, tau_d=tau_d)
system_compensated = DoubleIntegratorSystem(dt, tau_d=tau_d)  # Same delay but with compensated feedback

controller_no_delay = PDController(Kp, Kd, dt)
controller_with_delay = PDController(Kp, Kd, dt)
controller_compensated = PDController(Kp, Kd, dt)  # Controller for compensated system

lpf_with_delay = LowPassFilter(omega_net, dt)

diff_with_delay = Differentiator(dt)

# Initialize consistent integrator for disturbance
consistent_integrator = ConsistentIntegrator(dt)

# Storage arrays
x_no_delay = np.zeros(n_steps)
v_no_delay = np.zeros(n_steps)
u_no_delay = np.zeros(n_steps)

x_with_delay = np.zeros(n_steps)
x_delayed = np.zeros(n_steps)
v_with_delay = np.zeros(n_steps)
u_with_delay = np.zeros(n_steps)
disturbance_with_delay = np.zeros(n_steps)
filtered_disturbance_with_delay = np.zeros(n_steps)

# Additional arrays for double integration and compensation
double_integrated_disturbance = np.zeros(n_steps)
compensated_feedback = np.zeros(n_steps)

# Arrays for compensated system
x_compensated = np.zeros(n_steps)
x_delayed_compensated = np.zeros(n_steps)
v_compensated = np.zeros(n_steps)
u_compensated = np.zeros(n_steps)

print("Running simulation...")
print(f"PD Controller gains: Kp = {Kp}, Kd = {Kd}")
print(f"Output delay: {tau_d} seconds")
print(f"Low pass filter cutoff: {omega_net} rad/s")
print("\nCDOB Pipeline:")
print("1. Output from delayed system: x(t-τd)")
print("2. Control input: u(t) = α²*(xs-xf) + 2α*(xs_dot - xf_dot)")
print("3. Filtered disturbance: LPF(u(t) - ẍ(t-τd))")
print("4. Double integrate filtered disturbance and add to delayed output: xf = x(t-τd) + ∫∫filtered_disturbance")
print("5. Use compensated feedback xf in controller for next iteration")

# Simulation loop
for i in range(n_steps):
    # System without delay (no disturbance calculation needed)
    u_no_delay[i] = controller_no_delay.control(r[i], x_no_delay[i-1] if i > 0 else 0)
    x_no_delay[i], v_no_delay[i], _ = system_no_delay.step(u_no_delay[i])
    
    # System with delay (uncompensated) - for comparison only
    u_with_delay[i] = controller_with_delay.control(r[i], x_delayed[i-1] if i > 0 else 0)
    x_with_delay[i], v_with_delay[i], x_delayed[i] = system_with_delay.step(u_with_delay[i])
    
    # CDOB Pipeline Implementation:
    # Step 1: Get delayed output from compensated system
    if i > 0:
        # Step 2: Calculate control input using compensated feedback from previous step
        u_compensated[i] = controller_compensated.control(r[i], compensated_feedback[i-1])
    else:
        u_compensated[i] = controller_compensated.control(r[i], 0.0)
    
    # Apply control to system and get delayed output
    x_compensated[i], v_compensated[i], x_delayed_compensated[i] = system_compensated.step(u_compensated[i])
    
    # Step 3: Calculate disturbance using the compensated system's control input and delayed output
    x_ddot_with_delay = diff_with_delay.differentiate(x_delayed_compensated[i])
    disturbance_with_delay[i] = u_compensated[i] - x_ddot_with_delay
    filtered_disturbance_with_delay[i] = lpf_with_delay.filter(disturbance_with_delay[i])
      # Step 4: Double integrate the filtered disturbance using consistent RK45 method
    double_integrated_disturbance[i] = consistent_integrator.double_integrate(filtered_disturbance_with_delay[i])
    
    # Step 5: Create compensated feedback: delayed output + double integrated disturbance
    compensated_feedback[i] = x_delayed_compensated[i] + double_integrated_disturbance[i]

print("Simulation completed!")

# Plotting
plt.figure(figsize=(10, 6))

# Plot 1: Position response comparison
plt.subplot(3, 2, 1)
plt.plot(t, r, 'k--', label='Reference', linewidth=2)
plt.plot(t, x_no_delay, 'b-', label='No Delay', linewidth=2)
plt.plot(t, x_delayed, 'r-', label='With Delay (Uncompensated)', linewidth=2)
plt.plot(t, x_delayed_compensated, 'g-', label='With Delay (CDOB Compensated)', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Position Response Comparison')
plt.legend()
plt.xlim(0, 3)

# Plot 2: Control input
plt.subplot(3, 2, 2)
plt.plot(t, u_no_delay, 'b-', label='No Delay', linewidth=2)
plt.plot(t, u_with_delay, 'r-', label='With Delay (Uncompensated)', linewidth=2)
plt.plot(t, u_compensated, 'g-', label='With Delay (CDOB Compensated)', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Control Input (m/s²)')
plt.title('Control Input (Acceleration)')
plt.legend()
plt.xlim(0, 3)

# Plot 3: Raw disturbance (CDOB system)
plt.subplot(3, 2, 3)
plt.plot(t, disturbance_with_delay, 'g-', label='CDOB System', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Raw Disturbance')
plt.title('Raw Disturbance (u - ẍ) - CDOB System')
plt.legend()
plt.xlim(0, 3)

# Plot 4: Filtered disturbance (CDOB system)
plt.subplot(3, 2, 4)
plt.plot(t, filtered_disturbance_with_delay, 'g-', label='CDOB System', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Filtered Disturbance')
plt.title('Filtered Disturbance - CDOB System')
plt.legend()
plt.xlim(0, 3)

# Plot 5: Double integrated disturbance
plt.subplot(3, 2, 5)
plt.plot(t, double_integrated_disturbance, 'g-', label='Double Integrated', linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Double Integrated Disturbance (m)')
plt.title('Double Integrated Disturbance (Consistent RK45)')
plt.legend()
plt.xlim(0, 3)

# Plot 6: Compensation error
plt.subplot(3, 2, 6)
compensation_error = x_no_delay - x_delayed_compensated
uncompensated_error = x_no_delay - x_delayed
compensation_effectiveness = compensated_feedback - x_delayed_compensated
plt.plot(t, compensation_error, 'm-', label='No Delay - Compensated System', linewidth=2)
plt.plot(t, uncompensated_error, 'c--', label='No Delay - Uncompensated Delayed', linewidth=2, alpha=0.7)
plt.plot(t, compensation_effectiveness, 'orange', label='Compensated Feedback - Delayed Output', linewidth=1, alpha=0.8)
plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.title('Compensation Error Analysis')
plt.legend()
plt.xlim(0, 3)

plt.tight_layout()
plt.show()

# Optional plots (commented out but available)
def plot_optional():
    """Optional plots for velocity and acceleration"""
    plt.figure(figsize=(12, 8))
    
    # Velocity plot
    plt.subplot(2, 1, 1)
    plt.plot(t, v_no_delay, 'b-', label='No Delay', linewidth=2)
    plt.plot(t, v_with_delay, 'r-', label='With Delay', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Response')
    plt.legend()
    plt.xlim(0, 3)
    
    # Acceleration plot (same as control input for double integrator)
    plt.subplot(2, 1, 2)
    plt.plot(t, u_no_delay, 'b-', label='No Delay', linewidth=2)
    plt.plot(t, u_with_delay, 'r-', label='With Delay', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Input Acceleration')
    plt.legend()
    plt.xlim(0, 3)
    
    plt.tight_layout()
    plt.show()

# Uncomment the line below to show optional plots
# plot_optional()

print("\nSystem Analysis:")
print(f"Final position (no delay): {x_no_delay[-1]:.4f}")
print(f"Final position (with delay, uncompensated): {x_delayed[-1]:.4f}")
print(f"Final position (with delay, compensated system): {x_delayed_compensated[-1]:.4f}")
print(f"Final compensated feedback signal: {compensated_feedback[-1]:.4f}")
print(f"Final double integrated disturbance: {double_integrated_disturbance[-1]:.6f}")
print(f"\nRMS Errors:")
print(f"No delay vs compensated system: {np.sqrt(np.mean((x_no_delay - x_delayed_compensated)**2)):.6f}")
print(f"No delay vs uncompensated delayed: {np.sqrt(np.mean((x_no_delay - x_delayed)**2)):.6f}")
print(f"\nMax Absolute Errors:")
print(f"No delay vs compensated system: {np.max(np.abs(x_no_delay - x_delayed_compensated)):.6f}")
print(f"No delay vs uncompensated delayed: {np.max(np.abs(x_no_delay - x_delayed)):.6f}")
print(f"\nCompensation effectiveness: {(1 - np.sqrt(np.mean((x_no_delay - x_delayed_compensated)**2)) / np.sqrt(np.mean((x_no_delay - x_delayed)**2))) * 100:.2f}%")