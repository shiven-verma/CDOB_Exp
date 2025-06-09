import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SecondOrderIntegratorSystem:
    def __init__(self, alpha, tau_d=0.0, omega_net=0.0):
        self.alpha = alpha
        self.Kp = alpha**2  # P gain
        self.Kd = 2*alpha   # D gain
        self.tau_d = tau_d  # Output delay
        self.omega_net = omega_net  # Cutoff frequency for disturbance observer
        self.delay_buffer = []  # Buffer to store delayed outputs
        
        # Disturbance observer states
        self.lpf_state = 0.0  # Low pass filter state
        self.integrator1_state = 0.0  # First integrator state
        self.integrator2_state = 0.0  # Second integrator state (disturbance feedback)
        
    def system_dynamics(self, state, t, setpoint, t_span, dt):
        """
        Second order integrator: x'' = u
        State: [position, velocity]
        Control: u = Kp*(setpoint_corrected - position_delayed) - Kd*velocity_delayed
        """
        x, x_dot = state
        
        # Get delayed output for feedback
        x_delayed, x_dot_delayed = self.get_delayed_output(x, x_dot, t, t_span, dt)
        
        # Calculate double derivative of delayed position (numerical differentiation)
        x_ddot_delayed = self.get_delayed_acceleration(x_delayed, x_dot_delayed, t, dt)
          # Disturbance observer calculation (only if omega_net > 0 and there's delay)
        # When tau_d = 0 (no delay), disturbance_signal = 0 (no disturbance to compensate)
        if self.omega_net >= 0 and self.tau_d >= 0:
            # Step 1: Subtract double derivative of delayed position from control input
            # We need the control input without disturbance compensation first
            error_raw = setpoint - x
            u_raw = self.Kp * error_raw - self.Kd * x_dot
            
            # Step 2: Calculate disturbance signal
            disturbance_signal = u_raw - x_ddot_delayed

            # Step 3: Pass through low pass filter
            self.lpf_state = self.low_pass_filter(disturbance_signal, self.omega_net, dt)

            
            # Step 4: Pass filtered signal through double integrator with Simpson's method
            
            # Initialize history buffers for Simpson's integration
            if not hasattr(self, 'lpf_history'):
                self.lpf_history = [0.0, 0.0]  # Store [n-2, n-1] values
                self.integrator1_history = [0.0, 0.0]  # Store [n-2, n-1] values
                self.step_count = 0
            
            # First integrator: Use Simpson's rule for higher accuracy
            integrator1_input = self.lpf_state
            
            if self.step_count < 2:
                # Use Tustin for first two steps to initialize Simpson's
                if self.step_count == 0:
                    integrator1_delta = integrator1_input * dt  # Euler for first step
                else:
                    integrator1_delta = (dt / 2) * (integrator1_input + self.lpf_history[-1])
            else:
                # Simpson's rule: integral = (dt/3) * (f[n-2] + 4*f[n-1] + f[n])
                integrator1_delta = (dt / 3) * (self.lpf_history[0] + 4*self.lpf_history[1] + integrator1_input)
            
            # Anti-windup: Check if we're saturating before integration
            integrator1_temp = self.integrator1_state + integrator1_delta
            max_integrator1 = 100.0  # Saturation limit for stability
            
            if abs(integrator1_temp) <= max_integrator1:
                self.integrator1_state = integrator1_temp
            else:
                # Apply saturation with anti-windup
                self.integrator1_state = np.sign(integrator1_temp) * max_integrator1
            
            # Second integrator: Use Simpson's rule
            integrator2_input = self.integrator1_state
            
            if self.step_count < 2:
                # Use Tustin for first two steps
                if self.step_count == 0:
                    integrator2_delta = integrator2_input * dt  # Euler for first step
                else:
                    integrator2_delta = (dt / 2) * (integrator2_input + self.integrator1_history[-1])
            else:
                # Simpson's rule for second integrator
                integrator2_delta = (dt / 3) * (self.integrator1_history[0] + 4*self.integrator1_history[1] + integrator2_input)
            
            integrator2_temp = self.integrator2_state + integrator2_delta
            max_integrator2 = 100.0  # Saturation limit for stability
            
            if abs(integrator2_temp) <= max_integrator2:
                self.integrator2_state = integrator2_temp
            else:
                # Apply saturation with anti-windup
                self.integrator2_state = np.sign(integrator2_temp) * max_integrator2
            
            # Update history buffers for Simpson's method
            self.lpf_history = [self.lpf_history[1], integrator1_input]
            self.integrator1_history = [self.integrator1_history[1], integrator2_input]
            self.step_count += 1
              # Step 5: Add with delayed system output to get disturbance feedback
            disturbance_feedback = self.integrator2_state + x_delayed
            # if self.omega_net > 0 and self.tau_d > 0:
            #     # print("Disturbance feedback:", disturbance_feedback,"x_delayed:", x_delayed,'x:',x)
            #     print("Disturbance Signal:", disturbance_signal,"lpf_state:",self.lpf_state)
            
            # Step 6: Calculate disturbance feedback velocity (derivative of disturbance feedback)
            disturbance_feedback_velocity = self.get_disturbance_feedback_velocity(disturbance_feedback, dt)
            
            # Step 7: Use disturbance feedback as the feedback signal
            error = setpoint - disturbance_feedback
            feedback_velocity = disturbance_feedback_velocity
            

        
        # PD Controller using corrected feedback
        u = self.Kp * error - self.Kd * feedback_velocity
        
        # System dynamics: x'' = u
        x_ddot = u
        
        return [x_dot, x_ddot]
    
    def get_delayed_acceleration(self, x_delayed, x_dot_delayed, t, dt):
        """Calculate acceleration of delayed position using filtered numerical differentiation"""
        if not hasattr(self, 'prev_x_dot_delayed'):
            self.prev_x_dot_delayed = x_dot_delayed
            self.prev_x_ddot_delayed = 0.0
            return 0.0
        
        # Calculate raw derivative
        x_ddot_raw = (x_dot_delayed - self.prev_x_dot_delayed) / dt
          # Apply low-pass filter to derivative to reduce noise amplification
        # Using a simple first-order filter with cutoff frequency = 10 * omega_net
        if self.omega_net > 0:
            filter_cutoff = min(10.0 * self.omega_net, 100.0)  # Limit to reasonable value
            x_ddot_delayed = self.low_pass_filter_derivative(x_ddot_raw, filter_cutoff, dt)
        else:
            x_ddot_delayed = x_ddot_raw
        
        # Store for next iteration
        self.prev_x_dot_delayed = x_dot_delayed
        self.prev_x_ddot_delayed = x_ddot_delayed
        
        return x_ddot_delayed
    
    def get_delayed_output(self, x, x_dot, t, t_span, dt):
        """Get delayed output based on tau_d"""
        if self.tau_d == 0:
            return x, x_dot
        
        # Store current state in delay buffer
        self.delay_buffer.append([t, x, x_dot])
        
        # Limit buffer size to prevent memory issues (keep only recent history)
        max_buffer_size = max(1000, int(self.tau_d / (dt * 0.1)))  # Keep enough for interpolation
        if len(self.delay_buffer) > max_buffer_size:
            self.delay_buffer = self.delay_buffer[-max_buffer_size:]
        
        # Calculate delayed time
        t_delayed = t - self.tau_d
        
        if t_delayed <= t_span[0]:
            # If delayed time is before simulation start, return initial conditions
            return 0.0, 0.0
        
        # Find the closest stored state for the delayed time
        for i, (stored_t, stored_x, stored_x_dot) in enumerate(self.delay_buffer):
            if stored_t >= t_delayed:
                if i == 0:
                    return stored_x, stored_x_dot
                else:
                    # Linear interpolation between two points
                    t1, x1, x_dot1 = self.delay_buffer[i-1]
                    t2, x2, x_dot2 = self.delay_buffer[i]
                    alpha_interp = (t_delayed - t1) / (t2 - t1) if t2 != t1 else 0
                    x_interp = x1 + alpha_interp * (x2 - x1)
                    x_dot_interp = x_dot1 + alpha_interp * (x_dot2 - x_dot1)
                    return x_interp, x_dot_interp
        
        # If no suitable point found, return the last available state
        if self.delay_buffer:
            return self.delay_buffer[-1][1], self.delay_buffer[-1][2]
        else:
            return 0.0, 0.0

    def simulate(self, setpoint, t_span, initial_state=[0, 0]):
        """Simulate the system response to a step setpoint"""
        # Clear delay buffer for new simulation
        self.delay_buffer = []
        
        # Reset disturbance observer states
        self.lpf_state = 0.0
        self.integrator1_state = 0.0
        self.integrator2_state = 0.0
        
        # Reset Simpson's integration history buffers
        if hasattr(self, 'lpf_history'):
            delattr(self, 'lpf_history')
        if hasattr(self, 'integrator1_history'):
            delattr(self, 'integrator1_history')
        if hasattr(self, 'step_count'):
            delattr(self, 'step_count')        # Reset differentiation history
        if hasattr(self, 'prev_x_dot_delayed'):
            delattr(self, 'prev_x_dot_delayed')
        if hasattr(self, 'prev_x_ddot_delayed'):
            delattr(self, 'prev_x_ddot_delayed')
        if hasattr(self, 'deriv_filter_state'):
            delattr(self, 'deriv_filter_state')
        if hasattr(self, 'prev_disturbance_feedback'):
            delattr(self, 'prev_disturbance_feedback')
        
        # Calculate average dt from t_span for numerical differentiation
        dt = (t_span[-1] - t_span[0]) / (len(t_span) - 1) if len(t_span) > 1 else 0.01
        
        # Store reference to time span and dt for dynamics function
        self.current_t_span = t_span
        self.current_dt = dt
        
        # Solve ODE using solve_ivp with more robust settings
        def dynamics_wrapper(t, state):
            return self.system_dynamics(state, t, setpoint, self.current_t_span, self.current_dt)
        
        # Use more relaxed tolerances and smaller maximum step size for stability
        solution = solve_ivp(dynamics_wrapper, [t_span[0], t_span[-1]], initial_state, 
                           t_eval=t_span, method='RK45', rtol=1e-2, atol=1e-5, 
                           max_step=dt*2)  # Limit step size to be close to desired dt
        
        if not solution.success:
            print(f"Warning: ODE solver failed: {solution.message}")
            # Return zeros if solver fails
            return np.zeros((len(t_span), 2))
        
        return solution.y.T

    def low_pass_filter(self, input_signal, cutoff_freq, dt):
        """
        Low-pass filter that can handle zero cutoff frequency
        When cutoff_freq = 0, no signal passes through (output = 0)
        When cutoff_freq > 0, applies first-order low-pass filter
        """
        if cutoff_freq <= 0:
            # No signal passes through when cutoff frequency is zero or negative
            return 0.0
        
        # Standard first-order low-pass filter
        tau_lpf = 1.0 / cutoff_freq
        alpha_lpf = dt / (tau_lpf + dt)
        filtered_output = alpha_lpf * input_signal + (1 - alpha_lpf) * self.lpf_state
        
        return filtered_output
    
    def low_pass_filter_derivative(self, input_signal, cutoff_freq, dt):
        """
        Low-pass filter for derivative calculation
        Maintains its own state separate from the main LPF
        """
        if cutoff_freq <= 0:
            return input_signal  # No filtering if cutoff is zero
        
        if not hasattr(self, 'deriv_filter_state'):
            self.deriv_filter_state = 0.0
        
        tau_deriv_filter = 1.0 / cutoff_freq
        alpha_deriv = dt / (tau_deriv_filter + dt)
        filtered_output = alpha_deriv * input_signal + (1 - alpha_deriv) * self.deriv_filter_state
        
        # Update filter state
        self.deriv_filter_state = filtered_output
        
        return filtered_output
    
    def get_disturbance_feedback_velocity(self, disturbance_feedback, dt):
        """Calculate velocity of disturbance feedback using numerical differentiation"""
        if not hasattr(self, 'prev_disturbance_feedback'):
            self.prev_disturbance_feedback = disturbance_feedback
            return 0.0
        
        # Calculate derivative
        disturbance_feedback_velocity = (disturbance_feedback - self.prev_disturbance_feedback) / dt
        
        # Store for next iteration
        self.prev_disturbance_feedback = disturbance_feedback
        
        return disturbance_feedback_velocity
    
def main():
    # System parameters
    alpha = 5.0  # Controller parameter
    setpoint = 1.0  # Step input
    tau_d = 0.01  # Output delay in seconds
    omega_net = 45.0  # Cutoff frequency for disturbance observer
    
    # Time parameters
    T = 2.0  # Total simulation time (reduced for testing)
    dt = 0.01
    t = np.arange(0, T, dt)
    
    # Create systems for comparison
    system_no_delay = SecondOrderIntegratorSystem(alpha, tau_d=0.0, omega_net=0.0)
    system_with_delay = SecondOrderIntegratorSystem(alpha, tau_d=tau_d, omega_net=0.0)
    system_with_cdob = SecondOrderIntegratorSystem(alpha, tau_d=tau_d, omega_net=omega_net)
    
    # Simulate system responses
    initial_state = [0, 0]  # Start at rest at origin
    response_no_delay = system_no_delay.simulate(setpoint, t, initial_state)
    response_with_delay = system_with_delay.simulate(setpoint, t, initial_state)
    response_with_cdob = system_with_cdob.simulate(setpoint, t, initial_state)
    
    position_no_delay = response_no_delay[:, 0]
    velocity_no_delay = response_no_delay[:, 1]
    position_with_delay = response_with_delay[:, 0]
    velocity_with_delay = response_with_delay[:, 1]
    position_with_cdob = response_with_cdob[:, 0]
    velocity_with_cdob = response_with_cdob[:, 1]
    
    # Calculate control inputs for plotting
    error_no_delay = setpoint - position_no_delay
    control_input_no_delay = system_no_delay.Kp * error_no_delay - system_no_delay.Kd * velocity_no_delay
    
    error_with_delay = setpoint - position_with_delay
    control_input_with_delay = system_with_delay.Kp * error_with_delay - system_with_delay.Kd * velocity_with_delay
    
    error_with_cdob = setpoint - position_with_cdob
    control_input_with_cdob = system_with_cdob.Kp * error_with_cdob - system_with_cdob.Kd * velocity_with_cdob
    
    # Create setpoint array for plotting
    setpoint_array = np.full_like(t, setpoint)
    
    # Plotting
    plt.figure(figsize=(14, 10))
    
    # Position response
    plt.subplot(3, 1, 1)
    plt.plot(t, position_no_delay, 'b-', label='Position (No Delay)', linewidth=2)
    plt.plot(t, position_with_delay, 'r-', label=f'Position (τd={tau_d}s)', linewidth=2)
    plt.plot(t, position_with_cdob, 'g-', label=f'Position (CDOB, ωn={omega_net})', linewidth=2)
    plt.plot(t, setpoint_array, 'k--', label='Setpoint', linewidth=2)
    plt.ylabel('Position')
    plt.title(f'Second Order Integrator with PD Controller, Output Delay, and CDOB (α={alpha})')
    plt.legend()
    plt.grid(True)
    
    # Velocity response
    plt.subplot(3, 1, 2)
    plt.plot(t, velocity_no_delay, 'b-', label='Velocity (No Delay)', linewidth=2)
    plt.plot(t, velocity_with_delay, 'r-', label=f'Velocity (τd={tau_d}s)', linewidth=2)
    plt.plot(t, velocity_with_cdob, 'g-', label=f'Velocity (CDOB, ωn={omega_net})', linewidth=2)
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)
    
    # Control input
    plt.subplot(3, 1, 3)
    plt.plot(t, control_input_no_delay, 'b-', label='Control Input (No Delay)', linewidth=2)
    plt.plot(t, control_input_with_delay, 'r-', label=f'Control Input (τd={tau_d}s)', linewidth=2)
    plt.plot(t, control_input_with_cdob, 'g-', label=f'Control Input (CDOB, ωn={omega_net})', linewidth=2)
    plt.ylabel('Control Input')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print system info
    print(f"System Parameters:")
    print(f"α = {alpha}")
    print(f"Kp = {system_with_cdob.Kp}")
    print(f"Kd = {system_with_cdob.Kd}")
    print(f"Output Delay (τd) = {tau_d} seconds")
    print(f"CDOB Cutoff Frequency (ωn) = {omega_net} rad/s")
    print(f"Setpoint = {setpoint}")
    
    # Calculate settling times (within 2% of setpoint)
    settling_tolerance = 0.02 * setpoint
    
    # No delay case
    settling_mask_no_delay = np.abs(position_no_delay - setpoint) <= settling_tolerance
    if np.any(settling_mask_no_delay):
        settling_idx_no_delay = np.where(settling_mask_no_delay)[0]
        for i in range(len(settling_idx_no_delay)-1, -1, -1):
            if i == 0 or settling_idx_no_delay[i] - settling_idx_no_delay[i-1] > 1:
                settling_time_no_delay = t[settling_idx_no_delay[i]]
                break
        print(f"Settling time (2%, No Delay): {settling_time_no_delay:.2f} seconds")
    
    # With delay case
    settling_mask_with_delay = np.abs(position_with_delay - setpoint) <= settling_tolerance
    if np.any(settling_mask_with_delay):
        settling_idx_with_delay = np.where(settling_mask_with_delay)[0]
        for i in range(len(settling_idx_with_delay)-1, -1, -1):
            if i == 0 or settling_idx_with_delay[i] - settling_idx_with_delay[i-1] > 1:
                settling_time_with_delay = t[settling_idx_with_delay[i]]
                break
        print(f"Settling time (2%, With Delay): {settling_time_with_delay:.2f} seconds")
    
    # With CDOB case
    settling_mask_with_cdob = np.abs(position_with_cdob - setpoint) <= settling_tolerance
    if np.any(settling_mask_with_cdob):
        settling_idx_with_cdob = np.where(settling_mask_with_cdob)[0]
        for i in range(len(settling_idx_with_cdob)-1, -1, -1):
            if i == 0 or settling_idx_with_cdob[i] - settling_idx_with_cdob[i-1] > 1:
                settling_time_with_cdob = t[settling_idx_with_cdob[i]]
                break
        print(f"Settling time (2%, With CDOB): {settling_time_with_cdob:.2f} seconds")

if __name__ == "__main__":
    main()
