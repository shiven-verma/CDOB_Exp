import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class IntegrationTester:
    """Test different integration methods against known analytical solutions"""
    
    def __init__(self):
        pass
    
    def test_function_1(self, t):
        """Test function: f(t) = sin(2*t) + 0.5*cos(t)"""
        return np.sin(2*t) + 0.5*np.cos(t)
    
    def test_function_1_integral(self, t):
        """Analytical integral of test_function_1"""
        return -0.5*np.cos(2*t) + 0.5*np.sin(t)
    
    def test_function_2(self, t):
        """Test function: f(t) = 2*t*exp(-0.5*t)"""
        return 2*t*np.exp(-0.5*t)
    
    def test_function_2_integral(self, t):
        """Analytical integral of test_function_2"""
        return -4*np.exp(-0.5*t)*(t + 2)
    
    def euler_integration(self, f_values, dt):
        """Simple Euler integration: y[n] = y[n-1] + f[n-1] * dt"""
        result = np.zeros_like(f_values)
        for i in range(1, len(f_values)):
            result[i] = result[i-1] + f_values[i-1] * dt
        return result
    
    def tustin_integration(self, f_values, dt):
        """Tustin (Trapezoidal) integration: y[n] = y[n-1] + (dt/2) * (f[n] + f[n-1])"""
        result = np.zeros_like(f_values)
        for i in range(1, len(f_values)):
            result[i] = result[i-1] + (dt/2) * (f_values[i] + f_values[i-1])
        return result
    
    def tustin_integration_with_saturation(self, f_values, dt, max_value=100.0):
        """Tustin integration with anti-windup saturation"""
        result = np.zeros_like(f_values)
        for i in range(1, len(f_values)):
            delta = (dt/2) * (f_values[i] + f_values[i-1])
            temp = result[i-1] + delta
            
            # Apply saturation with anti-windup
            if abs(temp) <= max_value:
                result[i] = temp
            else:
                result[i] = np.sign(temp) * max_value
        return result
    
    def simpson_integration(self, f_values, dt):
        """Simpson's rule integration (more accurate for smooth functions)"""
        result = np.zeros_like(f_values)
        if len(f_values) < 3:
            return self.tustin_integration(f_values, dt)
        
        # Use Simpson's rule where possible
        for i in range(2, len(f_values)):
            if i == 2:
                # First two steps use trapezoidal
                result[1] = result[0] + (dt/2) * (f_values[1] + f_values[0])
                result[2] = result[1] + (dt/2) * (f_values[2] + f_values[1])
            else:
                # Simpson's rule: integral from x[i-2] to x[i] = (dt/3) * (f[i-2] + 4*f[i-1] + f[i])
                simpson_increment = (dt/3) * (f_values[i-2] + 4*f_values[i-1] + f_values[i])
                result[i] = result[i-2] + simpson_increment
        
        return result
    
    def solve_ivp_integration(self, f_func, t_span):
        """Use scipy's solve_ivp as reference"""
        def ode_func(t, y):
            return [f_func(t)]
        
        solution = solve_ivp(ode_func, [t_span[0], t_span[-1]], [0.0], 
                           t_eval=t_span, method='RK45', rtol=1e-8, atol=1e-10)
        return solution.y[0]
    
    def test_integration_accuracy(self, test_func, analytical_integral, t_span, test_name=""):
        """Test all integration methods against analytical solution"""
        dt = t_span[1] - t_span[0]
        f_values = test_func(t_span)
        
        # Calculate integrals using different methods
        euler_result = self.euler_integration(f_values, dt)
        tustin_result = self.tustin_integration(f_values, dt)
        tustin_sat_result = self.tustin_integration_with_saturation(f_values, dt)
        simpson_result = self.simpson_integration(f_values, dt)
        solve_ivp_result = self.solve_ivp_integration(test_func, t_span)
        
        # Analytical solution (offset to match initial condition = 0)
        analytical_result = analytical_integral(t_span) - analytical_integral(t_span[0])
        
        # Calculate errors
        euler_error = np.abs(euler_result - analytical_result)
        tustin_error = np.abs(tustin_result - analytical_result)
        tustin_sat_error = np.abs(tustin_sat_result - analytical_result)
        simpson_error = np.abs(simpson_result - analytical_result)
        solve_ivp_error = np.abs(solve_ivp_result - analytical_result)
        
        # Plot results
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Function values
        plt.subplot(3, 2, 1)
        plt.plot(t_span, f_values, 'k-', linewidth=2, label='f(t)')
        plt.title(f'{test_name} - Input Function')
        plt.ylabel('f(t)')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Integration results
        plt.subplot(3, 2, 2)
        plt.plot(t_span, analytical_result, 'k-', linewidth=3, label='Analytical', alpha=0.8)
        plt.plot(t_span, solve_ivp_result, 'b--', linewidth=2, label='solve_ivp (RK45)')
        plt.plot(t_span, euler_result, 'r:', linewidth=2, label='Euler')
        plt.plot(t_span, tustin_result, 'g-.', linewidth=2, label='Tustin')
        plt.plot(t_span, tustin_sat_result, 'm-', linewidth=1, label='Tustin + Saturation')
        plt.plot(t_span, simpson_result, 'c-', linewidth=1, label='Simpson')
        plt.title(f'{test_name} - Integration Results')
        plt.ylabel('Integral')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Errors (log scale)
        plt.subplot(3, 2, 3)
        plt.semilogy(t_span, euler_error + 1e-12, 'r:', linewidth=2, label='Euler Error')
        plt.semilogy(t_span, tustin_error + 1e-12, 'g-.', linewidth=2, label='Tustin Error')
        plt.semilogy(t_span, tustin_sat_error + 1e-12, 'm-', linewidth=1, label='Tustin + Sat Error')
        plt.semilogy(t_span, simpson_error + 1e-12, 'c-', linewidth=1, label='Simpson Error')
        plt.semilogy(t_span, solve_ivp_error + 1e-12, 'b--', linewidth=2, label='solve_ivp Error')
        plt.title(f'{test_name} - Integration Errors (Log Scale)')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Error comparison (linear scale, zoomed)
        plt.subplot(3, 2, 4)
        max_reasonable_error = np.percentile(np.concatenate([tustin_error, simpson_error]), 95)
        plt.plot(t_span, euler_error, 'r:', linewidth=2, label='Euler Error')
        plt.plot(t_span, tustin_error, 'g-.', linewidth=2, label='Tustin Error')
        plt.plot(t_span, tustin_sat_error, 'm-', linewidth=1, label='Tustin + Sat Error')
        plt.plot(t_span, simpson_error, 'c-', linewidth=1, label='Simpson Error')
        plt.plot(t_span, solve_ivp_error, 'b--', linewidth=2, label='solve_ivp Error')
        plt.ylim(0, min(max_reasonable_error * 2, np.max(tustin_error) * 1.1))
        plt.title(f'{test_name} - Integration Errors (Linear Scale)')
        plt.ylabel('Absolute Error')
        plt.legend()
        plt.grid(True)
        
        # Plot 5: Final error statistics
        plt.subplot(3, 2, 5)
        methods = ['Euler', 'Tustin', 'Tustin+Sat', 'Simpson', 'solve_ivp']
        final_errors = [euler_error[-1], tustin_error[-1], tustin_sat_error[-1], 
                       simpson_error[-1], solve_ivp_error[-1]]
        mean_errors = [np.mean(euler_error), np.mean(tustin_error), np.mean(tustin_sat_error),
                      np.mean(simpson_error), np.mean(solve_ivp_error)]
        
        x_pos = np.arange(len(methods))
        width = 0.35
        plt.bar(x_pos - width/2, final_errors, width, label='Final Error', alpha=0.7)
        plt.bar(x_pos + width/2, mean_errors, width, label='Mean Error', alpha=0.7)
        plt.yscale('log')
        plt.xlabel('Integration Method')
        plt.ylabel('Error (log scale)')
        plt.title(f'{test_name} - Error Summary')
        plt.xticks(x_pos, methods, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Computational stability test
        plt.subplot(3, 2, 6)
        # Test with larger time step to see stability
        dt_large = dt * 5
        t_span_coarse = np.arange(t_span[0], t_span[-1], dt_large)
        f_values_coarse = test_func(t_span_coarse)
        
        euler_coarse = self.euler_integration(f_values_coarse, dt_large)
        tustin_coarse = self.tustin_integration(f_values_coarse, dt_large)
        analytical_coarse = analytical_integral(t_span_coarse) - analytical_integral(t_span_coarse[0])
        
        plt.plot(t_span_coarse, analytical_coarse, 'k-', linewidth=3, label='Analytical', alpha=0.8)
        plt.plot(t_span_coarse, euler_coarse, 'r:', linewidth=2, label=f'Euler (dt={dt_large:.3f})')
        plt.plot(t_span_coarse, tustin_coarse, 'g-.', linewidth=2, label=f'Tustin (dt={dt_large:.3f})')
        plt.title(f'{test_name} - Stability Test (Large dt)')
        plt.ylabel('Integral')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print error statistics
        print(f"\n{test_name} - Error Analysis:")
        print(f"{'Method':<15} {'Final Error':<12} {'Mean Error':<12} {'Max Error':<12}")
        print("-" * 55)
        print(f"{'Euler':<15} {euler_error[-1]:<12.2e} {np.mean(euler_error):<12.2e} {np.max(euler_error):<12.2e}")
        print(f"{'Tustin':<15} {tustin_error[-1]:<12.2e} {np.mean(tustin_error):<12.2e} {np.max(tustin_error):<12.2e}")
        print(f"{'Tustin+Sat':<15} {tustin_sat_error[-1]:<12.2e} {np.mean(tustin_sat_error):<12.2e} {np.max(tustin_sat_error):<12.2e}")
        print(f"{'Simpson':<15} {simpson_error[-1]:<12.2e} {np.mean(simpson_error):<12.2e} {np.max(simpson_error):<12.2e}")
        print(f"{'solve_ivp':<15} {solve_ivp_error[-1]:<12.2e} {np.mean(solve_ivp_error):<12.2e} {np.max(solve_ivp_error):<12.2e}")
        
        return {
            'euler': {'result': euler_result, 'error': euler_error},
            'tustin': {'result': tustin_result, 'error': tustin_error},
            'tustin_sat': {'result': tustin_sat_result, 'error': tustin_sat_error},
            'simpson': {'result': simpson_result, 'error': simpson_error},
            'solve_ivp': {'result': solve_ivp_result, 'error': solve_ivp_error},
            'analytical': analytical_result
        }

def main():
    """Run integration method comparison tests"""
    tester = IntegrationTester()
    
    # Test parameters
    T = 5.0  # Total time
    dt = 0.01  # Time step
    t = np.arange(0, T, dt)
    
    print("Integration Methods Comparison Test")
    print("="*50)
    print(f"Time span: 0 to {T} seconds")
    print(f"Time step: {dt} seconds")
    print(f"Number of points: {len(t)}")
    
    # Test 1: Oscillatory function
    print("\nTest 1: Oscillatory Function - f(t) = sin(2t) + 0.5*cos(t)")
    results1 = tester.test_integration_accuracy(
        tester.test_function_1, 
        tester.test_function_1_integral, 
        t, 
        "Test 1: Oscillatory"
    )
    
    # Test 2: Exponential decay function
    print("\nTest 2: Exponential Function - f(t) = 2t*exp(-0.5t)")
    results2 = tester.test_integration_accuracy(
        tester.test_function_2, 
        tester.test_function_2_integral, 
        t, 
        "Test 2: Exponential"
    )
    
    # Test 3: Step response simulation (similar to CDOB scenario)
    print("\nTest 3: Step Response Simulation")
    
    def step_response_func(t):
        """Simulated step response derivative: d/dt[1 - exp(-5t)]"""
        return 5 * np.exp(-5 * t)
    
    def step_response_integral(t):
        """Analytical integral: 1 - exp(-5t)"""
        return 1 - np.exp(-5 * t)
    
    results3 = tester.test_integration_accuracy(
        step_response_func, 
        step_response_integral, 
        t, 
        "Test 3: Step Response"
    )
    
    # Recommendations based on results
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR CDOB IMPLEMENTATION:")
    print("="*70)
    print("1. TUSTIN INTEGRATION:")
    print("   - More stable than Euler, especially for larger time steps")
    print("   - Good balance between accuracy and computational cost")
    print("   - Recommended for real-time control applications")
    print()
    print("2. SATURATION & ANTI-WINDUP:")
    print("   - Essential for preventing integrator windup")
    print("   - Should be applied BEFORE integration, not after")
    print("   - Consider reducing saturation limits for better control")
    print()
    print("3. DERIVATIVE FILTERING:")
    print("   - Critical for numerical differentiation in CDOB")
    print("   - Low-pass filter with cutoff ≈ 10 × ωₙ")
    print("   - Helps reduce noise amplification")
    print()
    print("4. TIME STEP CONSIDERATIONS:")
    print("   - Smaller dt improves accuracy but increases computation")
    print("   - Tustin method more forgiving to larger dt than Euler")
    print("   - For CDOB: dt should be << 1/ωₙ for stability")

if __name__ == "__main__":
    main()
