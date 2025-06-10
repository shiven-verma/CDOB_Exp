"""
Multi-Agent Consensus Control System
Implementation of two cases:

Case 1: Standard consensus
X_ddot = alpha^2*K*(1*xs - X) + 2*alpha*K*X_dot

Case 2: Enhanced consensus with additional control term
X_ddot = beta*(alpha^2*K*(1*xs - X) + 2*alpha*K*X_dot) + (I-beta*K)*LPF((X_dot(t)-X_dot(t-tau_d))/tau_d)

Where:
- K is pinned Laplacian matrix
- xs is setpoint (=1 for all agents)
- LPF is low-pass filter with cutoff frequency omega_d
- I is identity matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class MultiAgentConsensus:
    def __init__(self, N=5, alpha=2.0, beta=2, tau_d=0.1, omega_d=10.0):
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.tau_d = tau_d
        self.omega_d = omega_d
        
        # Create pinned Laplacian matrix K
        self.K = self.create_pinned_laplacian()
        self.I = np.eye(N)  # Identity matrix
        
        # Setpoint vector (all agents should reach xs = 1)
        self.xs = np.ones(N)
        
        # Low-pass filter states for delay compensation term
        self.lpf_states = np.zeros(N)
        
        # Delay buffer for velocity history
        self.velocity_buffer = []
        
    def create_pinned_laplacian(self):
        """Create pinned Laplacian matrix K for ring topology"""
        # Adjacency matrix for ring topology
        # A = np.zeros((self.N, self.N))
        # for i in range(self.N):
        #     A[i, (i+1) % self.N] = 1  # Next neighbor
        #     A[i, (i-1) % self.N] = 1  # Previous neighbor
        
        # # Degree matrix
        # D = np.diag(np.sum(A, axis=1))
        
        # # Laplacian matrix
        # L = D - A
        
        # # Pin the first agent (leader)
        # L[0, 0] += 1

        K = np.array([[1, 0, 0, 0, 0],
                      [-1, 2, -1, 0, 0],
                      [-1, -1, 2, 0, 0],
                      [0, 0, -1, 1, 0],
                      [0, -1, 0, 0, 1]])
        
        return K
    
    def low_pass_filter(self, input_signal, dt):
        """Apply first-order low-pass filter: tau*y_dot + y = u"""
        tau = 1.0 / self.omega_d
        alpha = dt / (tau + dt)
        self.lpf_states = alpha * input_signal + (1 - alpha) * self.lpf_states
        return self.lpf_states
    
    def get_delayed_velocity(self, current_velocity, dt):
        """Get velocity from tau_d seconds ago"""
        # Add current velocity to buffer
        self.velocity_buffer.append(current_velocity.copy())
        
        # Calculate required buffer size
        delay_steps = int(self.tau_d / dt)
        
        # Keep buffer size manageable
        if len(self.velocity_buffer) > delay_steps + 10:
            self.velocity_buffer.pop(0)
        
        # Return delayed velocity or current if not enough history
        if len(self.velocity_buffer) > delay_steps:
            return self.velocity_buffer[-(delay_steps+1)]
        else:
            return current_velocity
    
    def case1_dynamics(self, t, state):
        """Case 1: Standard consensus dynamics"""
        X = state[:self.N]
        X_dot = state[self.N:]
        
        # X_ddot = alpha^2*K*(1*xs - X) + 2*alpha*K*X_dot
        X_ddot = (self.alpha**2 * self.K @ (self.xs - X) - 
                  2 * self.alpha * self.K @ X_dot)
        
        return np.concatenate([X_dot, X_ddot])
    
    def case2_dynamics(self, t, state, dt=0.001):
        """Case 2: Enhanced consensus with additional control term"""
        X = state[:self.N]
        X_dot = state[self.N:]
        
        # Standard consensus term
        consensus_term = (self.alpha**2 * self.K @ (self.xs - X) - 
                         2 * self.alpha * self.K @ X_dot)
        
        # Additional control term: (I-beta*K)*LPF((X_dot(t)-X_dot(t-tau_d))/tau_d)
        X_dot_delayed = self.get_delayed_velocity(X_dot, dt)
        velocity_difference = (X_dot - X_dot_delayed) / self.tau_d if self.tau_d > 0 else np.zeros(self.N)
        lpf_output = self.low_pass_filter(velocity_difference, dt)
        additional_term = (self.I - self.beta * self.K) @ lpf_output
        
        # Combined dynamics
        X_ddot = self.beta * consensus_term + additional_term
        
        return np.concatenate([X_dot, X_ddot])
    
    def simulate(self, case, t_span, initial_conditions):
        """Simulate the system for specified case"""
        if case == 1:
            dynamics = self.case1_dynamics
        elif case == 2:
            dynamics = self.case2_dynamics
        else:
            raise ValueError("Case must be 1 or 2")
        
        # Reset for clean simulation
        self.lpf_states = np.zeros(self.N)
        self.velocity_buffer = []
        
        # Solve differential equation
        sol = solve_ivp(dynamics, t_span, initial_conditions, 
                       method='RK45', dense_output=True, rtol=1e-8)
        
        return sol
    
    def plot_results(self, sol1, sol2, t_eval):
        """Plot comparison of both cases"""
        # Evaluate solutions at specified time points
        states1 = sol1.sol(t_eval)
        states2 = sol2.sol(t_eval)
        
        X1 = states1[:self.N, :]
        X2 = states2[:self.N, :]
        
        # Plot positions
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        for i in range(self.N):
            plt.plot(t_eval, X1[i, :], label=f'Agent {i+1}')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Target')
        plt.title('Case 1: Standard Consensus')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        for i in range(self.N):
            plt.plot(t_eval, X2[i, :], label=f'Agent {i+1}')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Target')
        plt.title('Case 2: Enhanced Consensus with Additional Control Term')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.legend()
        plt.grid(True)
          # Plot maximum deformation (maximum distance between any two agents)
        plt.subplot(2, 2, 3)
        # Calculate maximum deformation for each time step
        max_deform1 = np.zeros(len(t_eval))
        max_deform2 = np.zeros(len(t_eval))
        
        for t_idx in range(len(t_eval)):
            # For Case 1: find maximum distance between any two agents
            distances1 = []
            distances2 = []
            for i in range(self.N):
                for j in range(i+1, self.N):
                    distances1.append(abs(X1[i, t_idx] - X1[j, t_idx]))
                    distances2.append(abs(X2[i, t_idx] - X2[j, t_idx]))
            max_deform1[t_idx] = max(distances1) if distances1 else 0
            max_deform2[t_idx] = max(distances2) if distances2 else 0
        
        plt.semilogy(t_eval, max_deform1, 'b-', label='Case 1', linewidth=2)
        plt.semilogy(t_eval, max_deform2, 'r-', label='Case 2', linewidth=2)
        plt.title('Maximum Deformation')
        plt.xlabel('Time (s)')
        plt.ylabel('Max |Xi - Xj|')
        plt.legend()
        plt.grid(True)
        
        # Plot comparison
        plt.subplot(2, 2, 4)
        for i in range(self.N):
            plt.plot(t_eval, X1[i, :], 'b--', alpha=0.7, linewidth=1)
            plt.plot(t_eval, X2[i, :], 'r-', alpha=0.8, linewidth=1.5)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Target')
        plt.title('Comparison: Case 1 (dashed) vs Case 2 (solid)')
        plt.xlabel('Time (s)')
        plt.ylabel('Position')
        plt.legend(['Case 1 (all agents)', 'Case 2 (all agents)', 'Target'])
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    # System parameters
    N = 5           # Number of agents
    alpha = 5.0     # Control parameter (Kp = alpha^2 = 4, Kd = 2*alpha = 4)
    beta = 8     # Scaling factor for enhanced consensus
    tau_d = 0.01     # Communication delay (100ms)
    omega_d = 10.0  # Low-pass filter cutoff frequency (10 rad/s)
    
    # Create system
    system = MultiAgentConsensus(N, alpha, beta, tau_d, omega_d)
    
    # Initial conditions: [positions, velocities]
    # Random initial positions, zero initial velocities
    np.random.seed(42)
    X0 = np.zeros(N)  # Random positions between -2 and 2
    X_dot0 = np.zeros(N)              # Zero initial velocities
    initial_state = np.concatenate([X0, X_dot0])
    
    # Simulation time
    t_span = [0, 5]  # 5 seconds
    t_eval = np.linspace(0, 5, 1000)
    
    # Simulate both cases
    print("Simulating Case 1: Standard consensus...")
    sol1 = system.simulate(1, t_span, initial_state)
    
    print("Simulating Case 2: Enhanced consensus with additional control term...")
    sol2 = system.simulate(2, t_span, initial_state)
    
    # Plot results
    system.plot_results(sol1, sol2, t_eval)
    
    # Print system parameters
    print(f"\nSystem Parameters:")
    print(f"Number of agents (N): {N}")
    print(f"Control parameter (alpha): {alpha}")
    print(f"Proportional gain (Kp): {alpha**2}")
    print(f"Derivative gain (Kd): {2*alpha}")
    print(f"Enhancement factor (beta): {beta}")
    print(f"Communication delay (tau_d): {tau_d} s")
    print(f"LPF cutoff frequency (omega_d): {omega_d} rad/s")
    print(f"\nPinned Laplacian matrix K:")
    print(system.K)

if __name__ == "__main__":
    main()
