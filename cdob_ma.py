"""
Multi-Agent CDOB (Communication Disturbance Observer) Control System
6 Agents with Triangular Graph Topology and Output Delay Compensation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# System parameters
alpha = 5.0                   # PD controller parameter
Kp, Kd = alpha**2, 2*alpha    # PD gains (per agent)
tau_d = 0.01                 # Output delay (seconds)
omega_net = 100.0             # CDOB filter cutoff frequency (rad/s)
n_agents = 6                  # Number of agents

# Triangular graph Laplacian matrix (6x6)
# Agent 0 is the leader connected to agent 1
# Triangular topology: 0-1-2-3-4-5 with additional connections forming triangles
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

# Simulation parameters
t_end, dt = 5.0, 0.001
t = np.arange(0, t_end, dt)
n_steps = len(t)
xs = 1.0                      # Source setpoint (reference)

class MultiAgentDoubleIntegrator:
    """Multi-agent double integrator system with optional output delay"""
    def __init__(self, n_agents, dt, tau_d=0):
        self.n_agents = n_agents
        self.dt = dt
        self.delay_buffer_size = int(tau_d / dt) if tau_d > 0 else 0
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

class MultiAgentPDController:
    """Multi-agent PD Controller"""
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

class LowPassFilter:
    """First-order low pass filter for each agent"""
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
    """Numerical second derivative for each agent using finite differences"""
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
    """Double integrator for each agent using consistent RK45 method"""
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
    'ideal': MultiAgentDoubleIntegrator(n_agents, dt, tau_d=0),
    'delayed': MultiAgentDoubleIntegrator(n_agents, dt, tau_d=tau_d),
    'cdob': MultiAgentDoubleIntegrator(n_agents, dt, tau_d=tau_d)
}

controllers = {
    name: MultiAgentPDController(n_agents, Kp, Kd, dt, K, B) 
    for name in systems.keys()
}

# CDOB components
lpf = LowPassFilter(n_agents, omega_net, dt)
diff = MultiAgentDifferentiator(n_agents, dt)
integrator = MultiAgentConsistentIntegrator(n_agents, dt)

# Storage arrays
data = {name: {
    'x': np.zeros((n_steps, n_agents)),      # positions
    'v': np.zeros((n_steps, n_agents)),      # velocities  
    'u': np.zeros((n_steps, n_agents)),      # control inputs
    'x_delayed': np.zeros((n_steps, n_agents))  # delayed positions
} for name in systems.keys()}

# CDOB-specific arrays
cdob_data = {
    'disturbance': np.zeros((n_steps, n_agents)),
    'filtered_dist': np.zeros((n_steps, n_agents)),
    'integrated_dist': np.zeros((n_steps, n_agents)),
    'compensated_fb': np.zeros((n_steps, n_agents))
}

print("Running Multi-Agent CDOB simulation...")
print(f"Parameters: {n_agents} agents, Kp={Kp}, Kd={Kd}, τd={tau_d}s, ωnet={omega_net} rad/s")
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
    
    # CDOB system
    # Use compensated feedback from previous step
    prev_fb = cdob_data['compensated_fb'][i-1] if i > 0 else np.zeros(n_agents)
    data['cdob']['u'][i] = controllers['cdob'].control(xs, prev_fb)
    data['cdob']['x'][i], data['cdob']['v'][i], data['cdob']['x_delayed'][i] = systems['cdob'].step(data['cdob']['u'][i])
    
    # CDOB pipeline for each agent
    x_ddot = diff.differentiate(data['cdob']['x_delayed'][i])
    cdob_data['disturbance'][i] = data['cdob']['u'][i] - x_ddot
    cdob_data['filtered_dist'][i] = lpf.filter(cdob_data['disturbance'][i])
    cdob_data['integrated_dist'][i] = integrator.double_integrate(cdob_data['filtered_dist'][i])
    cdob_data['compensated_fb'][i] = data['cdob']['x_delayed'][i] + cdob_data['integrated_dist'][i]

print("Simulation completed!")

# Plotting functions for multi-agent results
def plot_responses_only():
    """Plot only the position responses for all three systems"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Ideal system response
    for agent in range(n_agents):
        axes[0].plot(t, data['ideal']['x'][:, agent], 
                    label=f'Agent {agent}', linewidth=2)
    axes[0].axhline(y=xs, color='k', linestyle='--', linewidth=2, label='Reference')
    axes[0].set_xlim(0, 3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position (m)')
    axes[0].set_title('Ideal System (No Delay)')
    axes[0].legend()
    
    # Delayed system response
    for agent in range(n_agents):
        axes[1].plot(t, data['delayed']['x_delayed'][:, agent], 
                    label=f'Agent {agent}', linewidth=2)
    axes[1].axhline(y=xs, color='k', linestyle='--', linewidth=2, label='Reference')
    axes[1].set_xlim(0, 3)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Position (m)')
    axes[1].set_title('Delayed System (Uncompensated)')
    axes[1].legend()
    
    # CDOB system response
    for agent in range(n_agents):
        axes[2].plot(t, data['cdob']['x_delayed'][:, agent], 
                    label=f'Agent {agent}', linewidth=2)
    axes[2].axhline(y=xs, color='k', linestyle='--', linewidth=2, label='Reference')
    axes[2].set_xlim(0, 3)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Position (m)')
    axes[2].set_title('CDOB Compensated System')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def plot_detailed_analysis():
    """Optional detailed analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Agent positions comparison
    for agent in range(n_agents):
        axes[0,0].plot(t, data['ideal']['x'][:, agent], 
                      label=f'Ideal Agent {agent}', alpha=0.7)
        axes[0,0].plot(t, data['delayed']['x_delayed'][:, agent], '--',
                      label=f'Delayed Agent {agent}', alpha=0.7)
        axes[0,0].plot(t, data['cdob']['x_delayed'][:, agent], ':',
                      label=f'CDOB Agent {agent}', linewidth=2)
    
    axes[0,0].axhline(y=xs, color='k', linestyle='-', linewidth=2, label='Reference')
    axes[0,0].set_xlim(0, 3)
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Position (m)')
    axes[0,0].set_title('Multi-Agent Position Responses')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Control inputs for all agents
    for agent in range(n_agents):
        axes[0,1].plot(t, data['ideal']['u'][:, agent], 
                      label=f'Ideal Agent {agent}', alpha=0.7)
        axes[0,1].plot(t, data['delayed']['u'][:, agent], '--',
                      label=f'Delayed Agent {agent}', alpha=0.7)
        axes[0,1].plot(t, data['cdob']['u'][:, agent], ':',
                      label=f'CDOB Agent {agent}', linewidth=2)
    
    axes[0,1].set_xlim(0, 3)
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Control Input (m/s²)')
    axes[0,1].set_title('Control Inputs')
    
    # CDOB disturbance processing (show average)
    avg_disturbance = np.mean(cdob_data['disturbance'], axis=1)
    avg_filtered = np.mean(cdob_data['filtered_dist'], axis=1)
    avg_integrated = np.mean(cdob_data['integrated_dist'], axis=1)
    
    axes[0,2].plot(t, avg_disturbance, 'orange', label='Avg Raw Disturbance', alpha=0.7)
    axes[0,2].plot(t, avg_filtered, 'purple', label='Avg Filtered', linewidth=2)
    axes[0,2].plot(t, avg_integrated, 'brown', label='Avg Integrated', linewidth=2)
    axes[0,2].set_xlim(0, 3)
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Average Disturbance')
    axes[0,2].set_title('CDOB Disturbance Processing')
    axes[0,2].legend()
    
    # Consensus error (spread among agents)
    consensus_ideal = np.max(data['ideal']['x'], axis=1) - np.min(data['ideal']['x'], axis=1)
    consensus_delayed = np.max(data['delayed']['x_delayed'], axis=1) - np.min(data['delayed']['x_delayed'], axis=1)
    consensus_cdob = np.max(data['cdob']['x_delayed'], axis=1) - np.min(data['cdob']['x_delayed'], axis=1)
    
    axes[1,0].plot(t, consensus_ideal, 'b-', label='Ideal', linewidth=2)
    axes[1,0].plot(t, consensus_delayed, 'r--', label='Delayed', linewidth=2)
    axes[1,0].plot(t, consensus_cdob, 'g:', label='CDOB', linewidth=2)
    axes[1,0].set_xlim(0, 3)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Agent Spread (m)')
    axes[1,0].set_title('Consensus Performance')
    axes[1,0].legend()
    
    # Tracking error for leader (agent 0)
    leader_error_ideal = xs - data['ideal']['x'][:, 0]
    leader_error_delayed = xs - data['delayed']['x_delayed'][:, 0]
    leader_error_cdob = xs - data['cdob']['x_delayed'][:, 0]
    
    axes[1,1].plot(t, leader_error_ideal, 'b-', label='Ideal', linewidth=2)
    axes[1,1].plot(t, leader_error_delayed, 'r--', label='Delayed', linewidth=2)
    axes[1,1].plot(t, leader_error_cdob, 'g:', label='CDOB', linewidth=2)
    axes[1,1].set_xlim(0, 3)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Leader Tracking Error (m)')
    axes[1,1].set_title('Leader (Agent 0) Tracking Error')
    axes[1,1].legend()
    
    # Compensation effect
    compensation_effect = np.mean(cdob_data['compensated_fb'] - data['cdob']['x_delayed'], axis=1)
    axes[1,2].plot(t, compensation_effect, 'orange', label='Avg Compensation', linewidth=2)
    axes[1,2].set_xlim(0, 3)
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_xlabel('Time (s)')
    axes[1,2].set_ylabel('Compensation Effect (m)')
    axes[1,2].set_title('Average CDOB Compensation Effect')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()

# Show main results (position responses only)
plot_responses_only()

# Performance analysis
print(f"\nMulti-Agent Performance Analysis:")
print(f"Final positions:")
for agent in range(n_agents):
    print(f"  Agent {agent}:")
    print(f"    Ideal: {data['ideal']['x'][-1, agent]:.4f}")
    print(f"    Delayed: {data['delayed']['x_delayed'][-1, agent]:.4f}")
    print(f"    CDOB: {data['cdob']['x_delayed'][-1, agent]:.4f}")

# RMS consensus error
consensus_rms_ideal = np.sqrt(np.mean((np.max(data['ideal']['x'], axis=1) - np.min(data['ideal']['x'], axis=1))**2))
consensus_rms_delayed = np.sqrt(np.mean((np.max(data['delayed']['x_delayed'], axis=1) - np.min(data['delayed']['x_delayed'], axis=1))**2))
consensus_rms_cdob = np.sqrt(np.mean((np.max(data['cdob']['x_delayed'], axis=1) - np.min(data['cdob']['x_delayed'], axis=1))**2))

# RMS tracking error for leader
leader_rms_ideal = np.sqrt(np.mean((xs - data['ideal']['x'][:, 0])**2))
leader_rms_delayed = np.sqrt(np.mean((xs - data['delayed']['x_delayed'][:, 0])**2))
leader_rms_cdob = np.sqrt(np.mean((xs - data['cdob']['x_delayed'][:, 0])**2))

consensus_improvement = (1 - consensus_rms_cdob / consensus_rms_delayed) * 100
leader_improvement = (1 - leader_rms_cdob / leader_rms_delayed) * 100

print(f"\nRMS Consensus Errors (agent spread):")
print(f"  Ideal: {consensus_rms_ideal:.6f}")
print(f"  Delayed: {consensus_rms_delayed:.6f}")
print(f"  CDOB: {consensus_rms_cdob:.6f}")
print(f"  Consensus Improvement: {consensus_improvement:.2f}%")

print(f"\nRMS Leader Tracking Errors:")
print(f"  Ideal: {leader_rms_ideal:.6f}")
print(f"  Delayed: {leader_rms_delayed:.6f}")
print(f"  CDOB: {leader_rms_cdob:.6f}")
print(f"  Leader Tracking Improvement: {leader_improvement:.2f}%")

# Optional detailed analysis plots (uncomment to show)
# plot_detailed_analysis()
