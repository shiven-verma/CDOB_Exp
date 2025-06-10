"""
CDOb (Communication Disturbance Observer) Control System
Double Integrator with Output Delay and PD Control
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System parameters
alpha = 5.0                   # PD controller parameter
Kp, Kd = alpha**2, 2*alpha    # PD gains
tau_d = 0.045                   # Output delay (seconds)
omega_net = 100.0              # CDOB filter cutoff frequency (rad/s)

# Simulation parameters
t_end, dt = 5.0, 0.001
t = np.arange(0, t_end, dt)
n_steps = len(t)
r = np.ones_like(t)           # Unit step reference

class DoubleIntegratorSystem:
    """Double integrator system with optional output delay"""
    def __init__(self, dt, tau_d=0):
        self.dt = dt
        self.delay_buffer_size = int(tau_d / dt) if tau_d > 0 else 0
        self.reset()
    
    def reset(self):
        self.state = np.array([0.0, 0.0])  # [position, velocity]
        self.delay_buffer = [0.0] * max(1, self.delay_buffer_size)
        
    def dynamics(self, t, state, u):
        """System dynamics: [x_dot, v_dot] = [v, u]"""
        x, v = state
        return [v, u]
        
    def step(self, u):
        """Step forward with RK45 integration"""
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, u),
            [0, self.dt],
            self.state,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        self.state = sol.y[:, -1]
        x, v = self.state
        
        # Apply output delay
        if self.delay_buffer_size > 0:
            self.delay_buffer.append(x)
            x_delayed = self.delay_buffer.pop(0)
        else:
            x_delayed = x
            
        return x, v, x_delayed

class PDController:
    """PD Controller with error derivative estimation"""
    def __init__(self, Kp, Kd, dt):
        self.Kp, self.Kd, self.dt = Kp, Kd, dt
        self.prev_error = 0.0
        
    def control(self, reference, measurement):
        error = reference - measurement
        error_dot = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.Kp * error + self.Kd * error_dot

class LowPassFilter:
    """First-order low pass filter"""
    def __init__(self, omega_c, dt):
        self.omega_c, self.dt = omega_c, dt
        self.y_prev = 0.0
        
    def filter(self, x):
        if self.omega_c == 0.0:
            return 0.0
        alpha = self.dt*self.omega_c / (1.0 + self.omega_c*self.dt)
        y = self.y_prev + alpha * (x - self.y_prev) 
        # y = self.y_prev + self.omega_c * (x - self.y_prev) * self.dt
        self.y_prev = y
        return y

class Differentiator:
    """Numerical second derivative using finite differences"""
    def __init__(self, dt):
        self.dt = dt
        self.history = []
        
    def differentiate(self, x):
        self.history.append(x)
        if len(self.history) < 3:
            return 0.0
        elif len(self.history) > 5:
            self.history.pop(0)
        
        # 3-point finite difference for second derivative
        if len(self.history) >= 3:
            return (self.history[-1] - 2*self.history[-2] + self.history[-3]) / (self.dt**2)
        return 0.0

class ConsistentIntegrator:
    """Double integrator using consistent RK45 method"""
    def __init__(self, dt):
        self.dt = dt
        self.state = np.array([0.0, 0.0])  # [velocity, position]
        
    def dynamics(self, t, state, acceleration):
        """Integration dynamics: [v_dot, p_dot] = [acceleration, velocity]"""
        velocity, position = state
        return [acceleration, velocity]
        
    def double_integrate(self, acceleration):
        """Double integrate using RK45 method"""
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, acceleration),
            [0, self.dt],
            self.state,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        self.state = sol.y[:, -1]
        return self.state[1]  # Return position

# Initialize systems
systems = {
    'ideal': DoubleIntegratorSystem(dt, tau_d=0),
    'delayed': DoubleIntegratorSystem(dt, tau_d=tau_d),
    'cdob': DoubleIntegratorSystem(dt, tau_d=tau_d)
}

controllers = {name: PDController(Kp, Kd, dt) for name in systems.keys()}
lpf = LowPassFilter(omega_net, dt)
diff = Differentiator(dt)
integrator = ConsistentIntegrator(dt)

# Storage arrays
data = {name: {
    'x': np.zeros(n_steps), 'v': np.zeros(n_steps), 'u': np.zeros(n_steps),
    'x_delayed': np.zeros(n_steps)
} for name in systems.keys()}

# CDOB-specific arrays
cdob_data = {
    'disturbance': np.zeros(n_steps),
    'filtered_dist': np.zeros(n_steps),
    'integrated_dist': np.zeros(n_steps),
    'compensated_fb': np.zeros(n_steps)
}

print("Running CDOB simulation...")
print(f"Parameters: Kp={Kp}, Kd={Kd}, τd={tau_d}s, ωnet={omega_net} rad/s")

# Main simulation loop
for i in range(n_steps):
    # Ideal system (no delay)
    data['ideal']['u'][i] = controllers['ideal'].control(r[i], data['ideal']['x'][i-1] if i > 0 else 0)
    data['ideal']['x'][i], data['ideal']['v'][i], data['ideal']['x_delayed'][i] = systems['ideal'].step(data['ideal']['u'][i])
    
    # Delayed system (uncompensated)
    data['delayed']['u'][i] = controllers['delayed'].control(r[i], data['delayed']['x_delayed'][i-1] if i > 0 else 0)
    data['delayed']['x'][i], data['delayed']['v'][i], data['delayed']['x_delayed'][i] = systems['delayed'].step(data['delayed']['u'][i])
    
    # CDOB system
    # Use compensated feedback from previous step
    fb_signal = cdob_data['compensated_fb'][i-1] if i > 0 else 0.0
    data['cdob']['u'][i] = controllers['cdob'].control(r[i], fb_signal)
    data['cdob']['x'][i], data['cdob']['v'][i], data['cdob']['x_delayed'][i] = systems['cdob'].step(data['cdob']['u'][i])
    
    # CDOB pipeline
    x_ddot = diff.differentiate(data['cdob']['x_delayed'][i])
    cdob_data['disturbance'][i] = data['cdob']['u'][i] - x_ddot
    cdob_data['filtered_dist'][i] = lpf.filter(cdob_data['disturbance'][i])
    cdob_data['integrated_dist'][i] = integrator.double_integrate(cdob_data['filtered_dist'][i])
    cdob_data['compensated_fb'][i] = data['cdob']['x_delayed'][i] + cdob_data['integrated_dist'][i]

print("Simulation completed!")

# Essential plots
def plot_essential():
    """Essential plots showing main results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Position response comparison
    axes[0,0].plot(t, r, 'k--', label='Reference', linewidth=2)
    axes[0,0].plot(t, data['ideal']['x'], 'b-', label='Ideal (No Delay)', linewidth=2)
    axes[0,0].plot(t, data['delayed']['x_delayed'], 'r-', label='Delayed (Uncompensated)', linewidth=2)
    axes[0,0].plot(t, data['cdob']['x_delayed'], 'g-', label='CDOB Compensated', linewidth=2)
    axes[0,0].set_xlim(0, 3)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Position (m)')
    axes[0,0].set_title('Position Response Comparison')
    axes[0,0].legend()
    
    # Control inputs
    axes[0,1].plot(t, data['ideal']['u'], 'b-', label='Ideal', linewidth=2)
    axes[0,1].plot(t, data['delayed']['u'], 'r-', label='Delayed', linewidth=2)
    axes[0,1].plot(t, data['cdob']['u'], 'g-', label='CDOB', linewidth=2)
    axes[0,1].set_xlim(0, 3)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Control Input (m/s²)')
    axes[0,1].set_title('Control Input Comparison')
    axes[0,1].legend()
    
    # CDOB compensation components
    axes[1,0].plot(t, cdob_data['disturbance'], 'orange', label='Raw Disturbance', alpha=0.7)
    axes[1,0].plot(t, cdob_data['filtered_dist'], 'purple', label='Filtered', linewidth=2)
    axes[1,0].plot(t, cdob_data['integrated_dist'], 'brown', label='Double Integrated', linewidth=2)
    axes[1,0].set_xlim(0, 3)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Disturbance Components')
    axes[1,0].set_title('CDOB Disturbance Processing')
    axes[1,0].legend()
    
    # Error analysis
    error_cdob = data['ideal']['x'] - data['cdob']['x_delayed']
    error_delayed = data['ideal']['x'] - data['delayed']['x_delayed']
    compensation_effect = cdob_data['compensated_fb'] - data['cdob']['x_delayed']
    
    axes[1,1].plot(t, error_cdob, 'g-', label='CDOB Error', linewidth=2)
    axes[1,1].plot(t, error_delayed, 'r--', label='Uncompensated Error', alpha=0.7)
    axes[1,1].plot(t, compensation_effect, 'orange', label='Compensation Effect', alpha=0.8)
    axes[1,1].set_xlim(0, 3)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Error (m)')
    axes[1,1].set_title('Compensation Error Analysis')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

# Detailed plots (optional)
def plot_detailed():
    """Detailed plots with all CDOB components"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # All previous plots in more detail
    # Position responses
    axes[0,0].plot(t, r, 'k--', label='Reference', linewidth=2)
    axes[0,0].plot(t, data['ideal']['x'], 'b-', label='Ideal', linewidth=2)
    axes[0,0].plot(t, data['delayed']['x_delayed'], 'r-', label='Delayed', linewidth=2)
    axes[0,0].plot(t, data['cdob']['x_delayed'], 'g-', label='CDOB', linewidth=2)
    axes[0,0].plot(t, cdob_data['compensated_fb'], 'm:', label='Compensated FB', alpha=0.8)
    axes[0,0].set_xlim(0, 3)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_title('Position & Feedback Signals')
    axes[0,0].legend()
    
    # Velocity responses
    axes[0,1].plot(t, data['ideal']['v'], 'b-', label='Ideal', linewidth=2)
    axes[0,1].plot(t, data['delayed']['v'], 'r-', label='Delayed', linewidth=2)
    axes[0,1].plot(t, data['cdob']['v'], 'g-', label='CDOB', linewidth=2)
    axes[0,1].set_xlim(0, 3)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_title('Velocity Responses')
    axes[0,1].legend()
    
    # Control inputs
    axes[1,0].plot(t, data['ideal']['u'], 'b-', label='Ideal', linewidth=2)
    axes[1,0].plot(t, data['delayed']['u'], 'r-', label='Delayed', linewidth=2)
    axes[1,0].plot(t, data['cdob']['u'], 'g-', label='CDOB', linewidth=2)
    axes[1,0].set_xlim(0, 3)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_title('Control Inputs')
    axes[1,0].legend()
    
    # Disturbance processing
    axes[1,1].plot(t, cdob_data['disturbance'], 'orange', label='Raw', alpha=0.7)
    axes[1,1].plot(t, cdob_data['filtered_dist'], 'purple', label='Filtered', linewidth=2)
    axes[1,1].set_xlim(0, 3)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_title('Disturbance Signals')
    axes[1,1].legend()
    
    # Integration and compensation
    axes[2,0].plot(t, cdob_data['integrated_dist'], 'brown', label='Integrated Dist.', linewidth=2)
    axes[2,0].plot(t, data['cdob']['x_delayed'], 'g--', label='Delayed Output', alpha=0.7)
    axes[2,0].plot(t, cdob_data['compensated_fb'], 'm-', label='Compensated FB', linewidth=2)
    axes[2,0].set_xlim(0, 3)
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].set_title('Compensation Components')
    axes[2,0].legend()
    
    # Error comparison
    error_cdob = data['ideal']['x'] - data['cdob']['x_delayed']
    error_delayed = data['ideal']['x'] - data['delayed']['x_delayed']
    axes[2,1].plot(t, error_cdob, 'g-', label='CDOB Error', linewidth=2)
    axes[2,1].plot(t, error_delayed, 'r--', label='Uncompensated Error', alpha=0.7)
    axes[2,1].set_xlim(0, 3)
    axes[2,1].grid(True, alpha=0.3)
    axes[2,1].set_title('Error Comparison')
    axes[2,1].legend()
    
    plt.tight_layout()
    plt.show()

# Show essential plots
plot_essential()
# plot_detailed()

# Performance analysis
rms_cdob = np.sqrt(np.mean((data['ideal']['x'] - data['cdob']['x_delayed'])**2))
rms_delayed = np.sqrt(np.mean((data['ideal']['x'] - data['delayed']['x_delayed'])**2))
improvement = (1 - rms_cdob / rms_delayed) * 100

print(f"\nPerformance Analysis:")
print(f"Final positions:")
print(f"  Ideal: {data['ideal']['x'][-1]:.4f}")
print(f"  Delayed: {data['delayed']['x_delayed'][-1]:.4f}")
print(f"  CDOB: {data['cdob']['x_delayed'][-1]:.4f}")
print(f"\nRMS Tracking Errors:")
print(f"  Uncompensated: {rms_delayed:.6f}")
print(f"  CDOB: {rms_cdob:.6f}")
print(f"  Improvement: {improvement:.2f}%")

# Uncomment for detailed plots
# plot_detailed()
