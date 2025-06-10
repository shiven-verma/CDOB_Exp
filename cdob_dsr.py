"""
Multi-Agent CDOB + DSR (Communication Disturbance Observer + Delayed Self Reinforcement) Control System
6 Agents with Triangular Graph Topology, Output Delay Compensation, and Enhanced Cohesion

This implementation combines:
1. CDOB: Communication Disturbance Observer for output delay compensation
2. DSR: Delayed Self Reinforcement for improved agent cohesion

The enhanced control law includes both delay compensation and cohesion improvement:
X_ddot = beta * (alpha^2*(xs*B - K*X) + 2*alpha*K*X_dot) + (I-beta*K)*LPF((X_dot(t)-X_dot(t-tau_d))/tau_d)

Where CDOB provides compensated feedback for X in the control law.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System parameters
alpha = 5.0                  # PD controller parameter
Kp, Kd = alpha**2, 2*alpha    # PD gains (per agent)
beta = 2                 # DSR enhancement factor (beta > 2 for stability)
tau_o = 0.1                # System output delay (seconds)
tau_d = 0.1                 # DSR delay (seconds) - different from output delay
omega_net = 100.0            # CDOB filter cutoff frequency (rad/s)
omega_dsr = 20.0             # DSR low-pass filter cutoff frequency (rad/s)
n_agents = 6                 # Number of agents

# Validate beta for stability
if beta < 2.0:
    print(f"Warning: beta={beta} < 2. Setting beta=2.0 for stability.")
    # beta = 2.1

# Triangular graph Laplacian matrix (6x6) - corrected based on reference DSR.py
# Using a pinned Laplacian structure for consensus control
K = np.array([
    [1, 0, 0,  0,  0,  0],   # Agent 0: not connected to anything
    [-1, 2, -1, 0,  0,  0],   # Agent 1: connected to 0, 2
    [-1, -1, 2, 0, 0,  0],   # Agent 2: connected to 0, 1
    [0, 0, -1,  1, 0, 0],   # Agent 3: connected to 2
    [0,  -1, 0, 0,  1, 0],   # Agent 4: connected to 1
    [0,  -1,  -1, -1, -1,  4]    # Agent 5: connected to 1,2,3,4
])

# Input matrix B (only agent 0 receives the reference)
B = np.array([1, 0, 0, 0, 0, 0]).reshape(6, 1)
I = np.eye(n_agents)  # Identity matrix

# Simulation parameters
t_end, dt = 3.0, 0.001
t = np.arange(0, t_end, dt)
n_steps = len(t)
xs = 1.0                      # Source setpoint (reference)

class MultiAgentDoubleIntegrator:
    """Multi-agent double integrator system with optional output delay"""
    def __init__(self, n_agents, dt, tau_o=0):
        self.n_agents = n_agents
        self.dt = dt
        self.delay_buffer_size = int(tau_o / dt) if tau_o > 0 else 0
        self.reset()
    
    def reset(self):
        self.states = np.zeros((2 * self.n_agents,))  # [x1, v1, x2, v2, ..., x6, v6]
        self.delay_buffers = [[0.0] * max(1, self.delay_buffer_size) for _ in range(self.n_agents)]
        
    def get_positions(self):
        """Extract positions from state vector"""
        return self.states[::2]  # Every even index
        
    def get_velocities(self):
        """Extract velocities from state vector"""
        return self.states[1::2]  # Every odd index
        
    def dynamics(self, t, state, u_vector):
        """Multi-agent system dynamics"""
        derivatives = np.zeros_like(state)
        
        for i in range(self.n_agents):
            pos_idx = 2 * i
            vel_idx = 2 * i + 1
            
            # Double integrator: x_dot = v, v_dot = u
            derivatives[pos_idx] = state[vel_idx]  # position derivative = velocity
            derivatives[vel_idx] = u_vector[i]      # velocity derivative = control input
            
        return derivatives
        
    def step(self, u_vector):
        """Step forward with RK45 integration"""
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, u_vector),
            [0, self.dt],
            self.states,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        self.states = sol.y[:, -1]
        positions = self.get_positions()
        velocities = self.get_velocities()
        
        # Apply output delay to each agent
        positions_delayed = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            if self.delay_buffer_size > 0:
                self.delay_buffers[i].append(positions[i])
                positions_delayed[i] = self.delay_buffers[i].pop(0)
            else:
                positions_delayed[i] = positions[i]
                
        return positions, velocities, positions_delayed

class MultiAgentDSRController:
    """Multi-agent DSR Controller (Enhanced Consensus with Delayed Self Reinforcement)"""
    def __init__(self, n_agents, alpha, dt, K_laplacian, beta, omega_dsr, tau_d):
        self.n_agents = n_agents
        self.alpha = alpha  # Use alpha instead of Kp/Kd
        self.dt = dt
        self.K = K_laplacian
        self.beta = beta
        self.omega_dsr = omega_dsr
        self.tau_d = tau_d  # DSR delay
        self.I = np.eye(n_agents)
        
        # Setpoint vector (all agents should reach xs = 1)
        self.xs = np.ones(n_agents)
        
        # State variables
        self.velocity_buffer = []
        self.dsr_lpf_states = np.zeros(n_agents)
        
    def dsr_low_pass_filter(self, input_signal):
        """Apply first-order low-pass filter for DSR"""
        if self.omega_dsr == 0.0:
            return np.zeros(self.n_agents)
        
        tau = 1.0 / self.omega_dsr
        alpha = self.dt / (tau + self.dt)
        self.dsr_lpf_states = alpha * input_signal + (1 - alpha) * self.dsr_lpf_states
        return self.dsr_lpf_states
    
    def get_delayed_velocity(self, current_velocity):
        """Get velocity from tau_d seconds ago for DSR"""
        # Add current velocity to buffer
        self.velocity_buffer.append(current_velocity.copy())
        
        # Calculate required buffer size based on DSR delay tau_d
        delay_steps = int(self.tau_d / self.dt)
        
        # Keep buffer size manageable
        if len(self.velocity_buffer) > delay_steps + 10:
            self.velocity_buffer.pop(0)
        
        # Return delayed velocity or current if not enough history
        if len(self.velocity_buffer) > delay_steps:
            return self.velocity_buffer[-(delay_steps+1)]
        else:
            return current_velocity
    
    def control(self, positions, velocities):
        """Compute DSR-enhanced control using consensus dynamics"""
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # Standard consensus term: alpha^2*K*(xs - X) - 2*alpha*K*X_dot
        consensus_term = (self.alpha**2 * self.K @ (self.xs - positions) - 
                         2 * self.alpha * self.K @ velocities)
        
        # DSR enhancement term: (I-beta*K)*LPF((X_dot(t)-X_dot(t-tau_d))/tau_d)
        velocity_delayed = self.get_delayed_velocity(velocities)
        velocity_difference = (velocities - velocity_delayed) / self.tau_d if self.tau_d > 0 else np.zeros(self.n_agents)
        lpf_output = self.dsr_low_pass_filter(velocity_difference)
        dsr_term = (self.I - self.beta * self.K) @ lpf_output
        
        # Combined DSR control law: X_ddot = beta * consensus_term + dsr_term
        u_vector = self.beta * consensus_term + dsr_term
        
        return u_vector

class MultiAgentCDOBDSRController:
    """Multi-agent CDOB+DSR Controller (Combined delay compensation and enhanced consensus)"""
    def __init__(self, n_agents, alpha, dt, K_laplacian, beta, omega_dsr, tau_d):
        self.n_agents = n_agents
        self.alpha = alpha  # Use alpha instead of Kp/Kd
        self.dt = dt
        self.K = K_laplacian
        self.beta = beta
        self.omega_dsr = omega_dsr
        self.tau_d = tau_d  # DSR delay (different from system output delay)
        self.I = np.eye(n_agents)
        
        # Setpoint vector (all agents should reach xs = 1)
        self.xs = np.ones(n_agents)
        
        # State variables
        self.velocity_buffer = []
        self.dsr_lpf_states = np.zeros(n_agents)
        
    def dsr_low_pass_filter(self, input_signal):
        """Apply first-order low-pass filter for DSR"""
        if self.omega_dsr == 0.0:
            return np.zeros(self.n_agents)
        
        tau = 1.0 / self.omega_dsr
        alpha = self.dt / (tau + self.dt)
        self.dsr_lpf_states = alpha * input_signal + (1 - alpha) * self.dsr_lpf_states
        return self.dsr_lpf_states
    
    def get_delayed_velocity(self, current_velocity):
        """Get velocity from tau_d seconds ago for DSR (not tau_o)"""
        # Add current velocity to buffer
        self.velocity_buffer.append(current_velocity.copy())
        
        # Calculate required buffer size based on DSR delay tau_d
        delay_steps = int(self.tau_d / self.dt)
        
        # Keep buffer size manageable
        if len(self.velocity_buffer) > delay_steps + 10:
            self.velocity_buffer.pop(0)
        
        # Return delayed velocity or current if not enough history
        if len(self.velocity_buffer) > delay_steps:
            return self.velocity_buffer[-(delay_steps+1)]
        else:
            return current_velocity        
    def control(self, compensated_positions, compensated_velocities):
        """Compute enhanced control using CDOB + DSR"""
        # compensated_positions and velocities are already CDOB-compensated
        compensated_positions = np.array(compensated_positions)
        compensated_velocities = np.array(compensated_velocities)
        
        # Standard consensus term: alpha^2*K*(xs - X) - 2*alpha*K*X_dot
        # Use CDOB-compensated positions and velocities
        consensus_term = (self.alpha**2 * self.K @ (self.xs - compensated_positions) - 
                         2 * self.alpha * self.K @ compensated_velocities)
        
        # DSR enhancement term: (I-beta*K)*LPF((X_dot(t)-X_dot(t-tau_d))/tau_d)
        velocity_delayed = self.get_delayed_velocity(compensated_velocities)
        velocity_difference = (compensated_velocities - velocity_delayed) / self.tau_d if self.tau_d > 0 else np.zeros(self.n_agents)
        lpf_output = self.dsr_low_pass_filter(velocity_difference)
        dsr_term = (self.I - self.beta * self.K) @ lpf_output
        
        # Combined CDOB+DSR control law: X_ddot = beta * consensus_term + dsr_term
        u_vector = self.beta * consensus_term + dsr_term
        
        return u_vector

class LowPassFilter:
    """First-order low pass filter for CDOB (each agent)"""
    def __init__(self, n_agents, omega_c, dt):
        self.n_agents = n_agents
        self.omega_c, self.dt = omega_c, dt
        self.y_prev = np.zeros(n_agents)
        
    def filter(self, x_vector):
        if self.omega_c == 0.0:
            return np.zeros(self.n_agents)
        
        alpha = self.dt * self.omega_c / (1.0 + self.omega_c * self.dt)
        y = self.y_prev + alpha * (x_vector - self.y_prev)
        self.y_prev = y
        return y

class MultiAgentDifferentiator:
    """Numerical second derivative for CDOB (each agent)"""
    def __init__(self, n_agents, dt):
        self.n_agents = n_agents
        self.dt = dt
        self.histories = [[] for _ in range(n_agents)]
        
    def differentiate(self, x_vector):
        derivatives = np.zeros(self.n_agents)
        
        for i in range(self.n_agents):
            self.histories[i].append(x_vector[i])
            if len(self.histories[i]) < 3:
                derivatives[i] = 0.0
            elif len(self.histories[i]) > 5:
                self.histories[i].pop(0)
            
            # 3-point finite difference for second derivative
            if len(self.histories[i]) >= 3:
                derivatives[i] = (self.histories[i][-1] - 2*self.histories[i][-2] + 
                                self.histories[i][-3]) / (self.dt**2)
        
        return derivatives

class MultiAgentConsistentIntegrator:
    """Double integrator for CDOB (each agent)"""
    def __init__(self, n_agents, dt):
        self.n_agents = n_agents
        self.dt = dt
        self.states = np.zeros((2 * n_agents,))  # [v1, p1, v2, p2, ..., v6, p6]
        
    def dynamics(self, t, state, acceleration_vector):
        """Integration dynamics for all agents"""
        derivatives = np.zeros_like(state)
        
        for i in range(self.n_agents):
            vel_idx = 2 * i
            pos_idx = 2 * i + 1
            
            # [v_dot, p_dot] = [acceleration, velocity]
            derivatives[vel_idx] = acceleration_vector[i]  # velocity derivative = acceleration
            derivatives[pos_idx] = state[vel_idx]          # position derivative = velocity
            
        return derivatives
        
    def double_integrate(self, acceleration_vector):
        """Double integrate for all agents using RK45 method"""
        sol = solve_ivp(
            lambda t, y: self.dynamics(t, y, acceleration_vector),
            [0, self.dt],
            self.states,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        self.states = sol.y[:, -1]
        # Return positions (every odd index)
        return self.states[1::2]

# Initialize multi-agent systems
systems = {
    'ideal': MultiAgentDoubleIntegrator(n_agents, dt, tau_o=0),
    'delayed': MultiAgentDoubleIntegrator(n_agents, dt, tau_o=tau_o),
    'dsr_only': MultiAgentDoubleIntegrator(n_agents, dt, tau_o=0),
    'cdob_only': MultiAgentDoubleIntegrator(n_agents, dt, tau_o=tau_o),
    'cdob_dsr': MultiAgentDoubleIntegrator(n_agents, dt, tau_o=tau_o)
}

# Controllers
class MultiAgentPDController:
    """Standard Multi-agent PD Controller"""
    def __init__(self, n_agents, Kp, Kd, dt, K_laplacian, B):
        self.n_agents = n_agents
        self.Kp, self.Kd, self.dt = Kp, Kd, dt
        self.K = K_laplacian
        self.B = B
        self.prev_errors = np.zeros(n_agents)
        
    def control(self, xs, measurements):
        """Compute control for all agents using consensus protocol"""
        # measurements is the vector of position measurements for all agents
        measurements = np.array(measurements).reshape(-1, 1)
        
        # Consensus error: K * x
        consensus_error = self.K @ measurements
        
        # Reference tracking error for agent 0 only
        reference_error = self.B * xs
        
        # Total position error
        position_error = -consensus_error + reference_error
        
        # Derivative of error (using finite differences)
        error_dot = (position_error.flatten() - self.prev_errors) / self.dt
        self.prev_errors = position_error.flatten()
        
        # PD control law
        u_vector = self.Kp * position_error.flatten() + self.Kd * error_dot
        
        return u_vector

controllers = {
    'ideal': MultiAgentPDController(n_agents, Kp, Kd, dt, K, B),
    'delayed': MultiAgentPDController(n_agents, Kp, Kd, dt, K, B),
    'dsr_only': MultiAgentDSRController(n_agents, alpha, dt, K, beta, omega_dsr, tau_d),
    'cdob_only': MultiAgentPDController(n_agents, Kp, Kd, dt, K, B),
    'cdob_dsr': MultiAgentCDOBDSRController(n_agents, alpha, dt, K, beta, omega_dsr, tau_d)
}

# CDOB components (for both CDOB-only and CDOB+DSR)
cdob_components = {
    'cdob_only': {
        'lpf': LowPassFilter(n_agents, omega_net, dt),
        'diff': MultiAgentDifferentiator(n_agents, dt),
        'integrator': MultiAgentConsistentIntegrator(n_agents, dt)
    },
    'cdob_dsr': {
        'lpf': LowPassFilter(n_agents, omega_net, dt),
        'diff': MultiAgentDifferentiator(n_agents, dt),
        'integrator': MultiAgentConsistentIntegrator(n_agents, dt)
    }
}

# Storage arrays
data = {name: {
    'x': np.zeros((n_steps, n_agents)),      # positions
    'v': np.zeros((n_steps, n_agents)),      # velocities  
    'u': np.zeros((n_steps, n_agents)),      # control inputs
    'x_delayed': np.zeros((n_steps, n_agents))  # delayed positions
} for name in systems.keys()}

# CDOB-specific arrays
cdob_data = {
    'cdob_only': {
        'disturbance': np.zeros((n_steps, n_agents)),
        'filtered_dist': np.zeros((n_steps, n_agents)),
        'integrated_dist': np.zeros((n_steps, n_agents)),
        'compensated_fb': np.zeros((n_steps, n_agents))
    },
    'cdob_dsr': {
        'disturbance': np.zeros((n_steps, n_agents)),
        'filtered_dist': np.zeros((n_steps, n_agents)),
        'integrated_dist': np.zeros((n_steps, n_agents)),
        'compensated_fb': np.zeros((n_steps, n_agents))
    }
}

print("Running Multi-Agent CDOB + DSR simulation...")
print(f"Parameters: {n_agents} agents, Kp={Kp}, Kd={Kd}, β={beta}")
print(f"System output delay τo={tau_o}s, DSR delay τd={tau_d}s")
print(f"ωnet={omega_net} rad/s, ωdsr={omega_dsr} rad/s")
print(f"Laplacian matrix K:\n{K}")

# Main simulation loop
for i in range(n_steps):
    # Ideal system (no delay)
    prev_x_ideal = data['ideal']['x'][i-1] if i > 0 else np.zeros(n_agents)
    data['ideal']['u'][i] = controllers['ideal'].control(xs, prev_x_ideal)
    data['ideal']['x'][i], data['ideal']['v'][i], data['ideal']['x_delayed'][i] = systems['ideal'].step(data['ideal']['u'][i])
      # Delayed system (uncompensated)
    prev_x_delayed = data['delayed']['x_delayed'][i-1] if i > 0 else np.zeros(n_agents)
    data['delayed']['u'][i] = controllers['delayed'].control(xs, prev_x_delayed)
    data['delayed']['x'][i], data['delayed']['v'][i], data['delayed']['x_delayed'][i] = systems['delayed'].step(data['delayed']['u'][i])
      # DSR-only system (no CDOB)
    prev_x_dsr_only = data['dsr_only']['x_delayed'][i-1] if i > 0 else np.zeros(n_agents)
    prev_v_dsr_only = data['dsr_only']['v'][i-1] if i > 0 else np.zeros(n_agents)
    data['dsr_only']['u'][i] = controllers['dsr_only'].control(prev_x_dsr_only, prev_v_dsr_only)
    data['dsr_only']['x'][i], data['dsr_only']['v'][i], data['dsr_only']['x_delayed'][i] = systems['dsr_only'].step(data['dsr_only']['u'][i])
    
    # CDOB-only system
    prev_fb_cdob = cdob_data['cdob_only']['compensated_fb'][i-1] if i > 0 else np.zeros(n_agents)
    data['cdob_only']['u'][i] = controllers['cdob_only'].control(xs, prev_fb_cdob)
    data['cdob_only']['x'][i], data['cdob_only']['v'][i], data['cdob_only']['x_delayed'][i] = systems['cdob_only'].step(data['cdob_only']['u'][i])
    
    # CDOB pipeline for CDOB-only
    x_ddot_cdob = cdob_components['cdob_only']['diff'].differentiate(data['cdob_only']['x_delayed'][i])
    cdob_data['cdob_only']['disturbance'][i] = data['cdob_only']['u'][i] - x_ddot_cdob
    cdob_data['cdob_only']['filtered_dist'][i] = cdob_components['cdob_only']['lpf'].filter(cdob_data['cdob_only']['disturbance'][i])
    cdob_data['cdob_only']['integrated_dist'][i] = cdob_components['cdob_only']['integrator'].double_integrate(cdob_data['cdob_only']['filtered_dist'][i])
    cdob_data['cdob_only']['compensated_fb'][i] = data['cdob_only']['x_delayed'][i] + cdob_data['cdob_only']['integrated_dist'][i]
      # CDOB + DSR system
    prev_fb_dsr = cdob_data['cdob_dsr']['compensated_fb'][i-1] if i > 0 else np.zeros(n_agents)
    prev_v_dsr = data['cdob_dsr']['v'][i-1] if i > 0 else np.zeros(n_agents)
    
    # Enhanced control with DSR using CDOB-compensated feedback
    data['cdob_dsr']['u'][i] = controllers['cdob_dsr'].control(prev_fb_dsr, prev_v_dsr)
    data['cdob_dsr']['x'][i], data['cdob_dsr']['v'][i], data['cdob_dsr']['x_delayed'][i] = systems['cdob_dsr'].step(data['cdob_dsr']['u'][i])
    
    # CDOB pipeline for CDOB+DSR
    x_ddot_dsr = cdob_components['cdob_dsr']['diff'].differentiate(data['cdob_dsr']['x_delayed'][i])
    cdob_data['cdob_dsr']['disturbance'][i] = data['cdob_dsr']['u'][i] - x_ddot_dsr
    cdob_data['cdob_dsr']['filtered_dist'][i] = cdob_components['cdob_dsr']['lpf'].filter(cdob_data['cdob_dsr']['disturbance'][i])
    cdob_data['cdob_dsr']['integrated_dist'][i] = cdob_components['cdob_dsr']['integrator'].double_integrate(cdob_data['cdob_dsr']['filtered_dist'][i])
    cdob_data['cdob_dsr']['compensated_fb'][i] = data['cdob_dsr']['x_delayed'][i] + cdob_data['cdob_dsr']['integrated_dist'][i]

print("Simulation completed!")

# Enhanced plotting functions
def plot_responses_comparison():
    """Plot position responses comparing all five systems"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    systems_to_plot = ['ideal', 'delayed', 'dsr_only', 'cdob_only', 'cdob_dsr']
    titles = ['Ideal System (No Delay)', 'Delayed System (Uncompensated)', 
              'DSR Only', 'CDOB Only', 'CDOB + DSR']
    
    # Plot first 5 systems in a 2x3 grid
    for idx, (system, title) in enumerate(zip(systems_to_plot, titles)):
        if idx < 5:  # We have 5 systems to plot
            row, col = idx // 3, idx % 3
            if idx == 4:  # Last plot goes to position (1,2)
                row, col = 1, 2
            
            for agent in range(n_agents):
                if system == 'ideal':
                    axes[row, col].plot(t, data[system]['x'][:, agent], 
                                      label=f'Agent {agent}', linewidth=2)
                else:
                    axes[row, col].plot(t, data[system]['x_delayed'][:, agent], 
                                      label=f'Agent {agent}', linewidth=2)
            
            axes[row, col].axhline(y=xs, color='k', linestyle='--', linewidth=2, label='Reference')
            # axes[row, col].set_xlim(0, 3)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col].set_ylabel('Position (m)')
            axes[row, col].set_title(title)
            axes[row, col].legend()
    
    # Hide the empty subplot
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_dsr_only_analysis():
    """Plot focused analysis on DSR-only system"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # DSR vs Standard delayed system comparison
    for agent in range(n_agents):
        axes[0, 0].plot(t, data['delayed']['x_delayed'][:, agent], '--',
                       alpha=0.7, color=f'C{agent}', label=f'Delayed Agent {agent}')
        axes[0, 0].plot(t, data['dsr_only']['x_delayed'][:, agent], '-',
                       linewidth=2, color=f'C{agent}', label=f'DSR Agent {agent}')
    
    axes[0, 0].axhline(y=xs, color='k', linestyle=':', linewidth=2, label='Reference')
    axes[0, 0].set_xlim(0, 3)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].set_title('DSR vs Delayed System Comparison')
    axes[0, 0].legend()
    
    # Consensus performance
    delayed_spread = np.max(data['delayed']['x_delayed'], axis=1) - np.min(data['delayed']['x_delayed'], axis=1)
    dsr_spread = np.max(data['dsr_only']['x_delayed'], axis=1) - np.min(data['dsr_only']['x_delayed'], axis=1)
    
    axes[0, 1].plot(t, delayed_spread, 'r--', label='Delayed System', linewidth=2)
    axes[0, 1].plot(t, dsr_spread, 'orange', label='DSR Only', linewidth=2)
    axes[0, 1].set_xlim(0, 3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Agent Spread (m)')
    axes[0, 1].set_title('DSR Consensus Enhancement')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Control effort comparison
    delayed_control = np.mean(np.abs(data['delayed']['u']), axis=1)
    dsr_control = np.mean(np.abs(data['dsr_only']['u']), axis=1)
    
    axes[1, 0].plot(t, delayed_control, 'r--', label='Delayed System', linewidth=2)
    axes[1, 0].plot(t, dsr_control, 'orange', label='DSR Only', linewidth=2)
    axes[1, 0].set_xlim(0, 3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Average Control Effort')
    axes[1, 0].set_title('Control Effort: DSR vs Delayed')
    axes[1, 0].legend()
    
    # Leader tracking error
    delayed_leader_error = np.abs(xs - data['delayed']['x_delayed'][:, 0])
    dsr_leader_error = np.abs(xs - data['dsr_only']['x_delayed'][:, 0])
    
    axes[1, 1].plot(t, delayed_leader_error, 'r--', label='Delayed System', linewidth=2)
    axes[1, 1].plot(t, dsr_leader_error, 'orange', label='DSR Only', linewidth=2)
    axes[1, 1].set_xlim(0, 3)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Leader Tracking Error (m)')
    axes[1, 1].set_title('Leader Tracking: DSR vs Delayed')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_cohesion_analysis():
    """Plot cohesion analysis showing agent spread and convergence"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Calculate consensus metrics
    systems_to_analyze = ['ideal', 'delayed', 'dsr_only', 'cdob_only', 'cdob_dsr']
    colors = ['blue', 'red', 'orange', 'green', 'purple']
    labels = ['Ideal', 'Delayed', 'DSR Only', 'CDOB Only', 'CDOB + DSR']
    
    # Agent spread (max - min position)
    for i, (system, color, label) in enumerate(zip(systems_to_analyze, colors, labels)):
        if system == 'ideal':
            positions = data[system]['x']
        else:
            positions = data[system]['x_delayed']
        
        spread = np.max(positions, axis=1) - np.min(positions, axis=1)
        axes[0, 0].plot(t, spread, color=color, label=label, linewidth=2)
    
    axes[0, 0].set_xlim(0, 3)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Agent Spread (m)')
    axes[0, 0].set_title('Consensus Performance (Agent Spread)')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Leader tracking error
    for i, (system, color, label) in enumerate(zip(systems_to_analyze, colors, labels)):
        if system == 'ideal':
            leader_error = np.abs(xs - data[system]['x'][:, 0])
        else:
            leader_error = np.abs(xs - data[system]['x_delayed'][:, 0])
        
        axes[0, 1].plot(t, leader_error, color=color, label=label, linewidth=2)
    
    axes[0, 1].set_xlim(0, 3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Leader Tracking Error (m)')
    axes[0, 1].set_title('Leader (Agent 0) Tracking Performance')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Control effort comparison
    for i, (system, color, label) in enumerate(zip(systems_to_analyze, colors, labels)):
        control_effort = np.mean(np.abs(data[system]['u']), axis=1)
        axes[1, 0].plot(t, control_effort, color=color, label=label, linewidth=2)
    
    axes[1, 0].set_xlim(0, 3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Average Control Effort')
    axes[1, 0].set_title('Control Effort Comparison')
    axes[1, 0].legend()
    
    # DSR enhancement visualization
    if 'cdob_dsr' in systems_to_analyze:
        dsr_effect = np.mean(cdob_data['cdob_dsr']['compensated_fb'] - data['cdob_dsr']['x_delayed'], axis=1)
        cdob_effect = np.mean(cdob_data['cdob_only']['compensated_fb'] - data['cdob_only']['x_delayed'], axis=1)
        
        axes[1, 1].plot(t, cdob_effect, 'g-', label='CDOB Effect', linewidth=2)
        axes[1, 1].plot(t, dsr_effect, 'purple', label='CDOB + DSR Effect', linewidth=2)
        axes[1, 1].set_xlim(0, 3)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Compensation Effect (m)')
        axes[1, 1].set_title('DSR Enhancement Effect')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

# Show results
plot_responses_comparison()
# plot_dsr_only_analysis()
# plot_cohesion_analysis()

# Performance analysis
print(f"\nMulti-Agent CDOB + DSR Performance Analysis:")
print(f"Final positions:")
for agent in range(n_agents):
    print(f"  Agent {agent}:")
    print(f"    Ideal: {data['ideal']['x'][-1, agent]:.4f}")
    print(f"    Delayed: {data['delayed']['x_delayed'][-1, agent]:.4f}")
    print(f"    DSR Only: {data['dsr_only']['x_delayed'][-1, agent]:.4f}")
    print(f"    CDOB Only: {data['cdob_only']['x_delayed'][-1, agent]:.4f}")
    print(f"    CDOB + DSR: {data['cdob_dsr']['x_delayed'][-1, agent]:.4f}")

# Calculate performance metrics
systems_to_analyze = ['ideal', 'delayed', 'dsr_only', 'cdob_only', 'cdob_dsr']
print(f"\nRMS Consensus Errors (agent spread):")
for system in systems_to_analyze:
    if system == 'ideal':
        positions = data[system]['x']
    else:
        positions = data[system]['x_delayed']
    
    spread = np.max(positions, axis=1) - np.min(positions, axis=1)
    rms_spread = np.sqrt(np.mean(spread**2))
    print(f"  {system.capitalize()}: {rms_spread:.6f}")

print(f"\nRMS Leader Tracking Errors:")
for system in systems_to_analyze:
    if system == 'ideal':
        leader_error = xs - data[system]['x'][:, 0]
    else:
        leader_error = xs - data[system]['x_delayed'][:, 0]
    
    rms_leader = np.sqrt(np.mean(leader_error**2))
    print(f"  {system.capitalize()}: {rms_leader:.6f}")

print(f"\nControl Effort (RMS):")
for system in systems_to_analyze:
    control_rms = np.sqrt(np.mean(data[system]['u']**2))
    print(f"  {system.capitalize()}: {control_rms:.6f}")

# Improvement calculations
delayed_spread_rms = np.sqrt(np.mean((np.max(data['delayed']['x_delayed'], axis=1) - np.min(data['delayed']['x_delayed'], axis=1))**2))
dsr_spread_rms = np.sqrt(np.mean((np.max(data['dsr_only']['x_delayed'], axis=1) - np.min(data['dsr_only']['x_delayed'], axis=1))**2))
cdob_spread_rms = np.sqrt(np.mean((np.max(data['cdob_only']['x_delayed'], axis=1) - np.min(data['cdob_only']['x_delayed'], axis=1))**2))
cdob_dsr_spread_rms = np.sqrt(np.mean((np.max(data['cdob_dsr']['x_delayed'], axis=1) - np.min(data['cdob_dsr']['x_delayed'], axis=1))**2))

dsr_improvement = (1 - dsr_spread_rms / delayed_spread_rms) * 100
cdob_improvement = (1 - cdob_spread_rms / delayed_spread_rms) * 100
cdob_dsr_improvement = (1 - cdob_dsr_spread_rms / delayed_spread_rms) * 100
dsr_over_cdob = (1 - cdob_dsr_spread_rms / cdob_spread_rms) * 100

print(f"\nImprovement Analysis:")
print(f"  DSR improvement over delayed: {dsr_improvement:.2f}%")
print(f"  CDOB improvement over delayed: {cdob_improvement:.2f}%")
print(f"  CDOB+DSR improvement over delayed: {cdob_dsr_improvement:.2f}%")
print(f"  CDOB+DSR improvement over CDOB alone: {dsr_over_cdob:.2f}%")
