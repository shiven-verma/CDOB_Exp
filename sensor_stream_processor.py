import numpy as np
import matplotlib.pyplot as plt
import time
from lpf import lpf, sci_lowpass

class ModelSensor:
    def __init__(self, sampling_freq=100, noise_std=0.1):
        self.sampling_freq = sampling_freq
        self.dt = 1.0 / sampling_freq
        self.noise_std = noise_std
        self.time_start = time.time()
        
    def get_acceleration_data(self, t):
        """Simulate acceleration data with noise"""
        # Simulated acceleration signal: sine wave + noise
        signal = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)
        noise = np.random.normal(0, self.noise_std)
        return signal + noise

def double_integrate(data, dt):
    """Perform double integration: acceleration -> velocity -> position"""
    velocity = np.cumsum(data) * dt
    position = np.cumsum(velocity) * dt
    return velocity, position

def stream_and_process(duration=10, sampling_freq=100, omega_lpf=10):
    """Stream data, filter, integrate, and plot results"""
    
    sensor = ModelSensor(sampling_freq=sampling_freq)
    dt = 1.0 / sampling_freq
    num_samples = int(duration * sampling_freq)
    
    # Data storage
    time_data = []
    raw_data = []
    filtered_data = []
    
    print(f"Streaming data for {duration} seconds at {sampling_freq} Hz...")
    
    # Simulate streaming data
    for i in range(num_samples):
        current_time = i * dt
        
        # Get sensor data
        accel = sensor.get_acceleration_data(current_time)
        
        time_data.append(current_time)
        raw_data.append(accel)
        
        # Apply low pass filter (need enough samples for filtering)
        if len(raw_data) >= 10:
            # Filter the current window of data
            window_data = np.array(raw_data[-10:])
            filtered_window = lpf(omega_lpf, len(window_data), window_data, dt)
            # filtered_window = sci_lowpass(omega_lpf, window_data, dt)
            filtered_data.append(filtered_window[-1])
        else:
            filtered_data.append(accel)
    
    # Convert to numpy arrays
    time_data = np.array(time_data)
    raw_data = np.array(raw_data)
    filtered_data = np.array(filtered_data)
    
    # Perform double integration on filtered data
    velocity, position = double_integrate(filtered_data, dt)
    
    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Raw vs Filtered Acceleration
    axes[0].plot(time_data, raw_data, alpha=0.7, label='Raw Acceleration', color='blue')
    axes[0].plot(time_data, filtered_data, label='Filtered Acceleration', color='red', linewidth=2)
    axes[0].set_ylabel('Acceleration (m/s²)')
    axes[0].set_title('Acceleration Data - Raw vs Filtered')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Velocity
    axes[1].plot(time_data, velocity, label='Velocity (integrated)', color='green', linewidth=2)
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].set_title('Velocity (Single Integration)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Position
    axes[2].plot(time_data, position, label='Position (double integrated)', color='purple', linewidth=2)
    axes[2].set_ylabel('Position (m)')
    axes[2].set_title('Position (Double Integration)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # All signals normalized for comparison
    axes[3].plot(time_data, raw_data/np.max(np.abs(raw_data)), alpha=0.5, label='Raw Accel (norm)')
    axes[3].plot(time_data, filtered_data/np.max(np.abs(filtered_data)), label='Filtered Accel (norm)')
    axes[3].plot(time_data, velocity/np.max(np.abs(velocity)), label='Velocity (norm)')
    axes[3].plot(time_data, position/np.max(np.abs(position)), label='Position (norm)')
    axes[3].set_ylabel('Normalized Amplitude')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('All Signals (Normalized)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return time_data, raw_data, filtered_data, velocity, position

def main():
    # Parameters
    duration = 10  # seconds
    sampling_freq = 100  # Hz
    omega_lpf = 5  # Low pass filter cutoff frequency
    
    print("Starting sensor data streaming and processing...")
    time_data, raw_data, filtered_data, velocity, position = stream_and_process(
        duration=duration, 
        sampling_freq=sampling_freq, 
        omega_lpf=omega_lpf
    )
    
    print(f"Processed {len(time_data)} samples")
    print(f"Final position: {position[-1]:.3f} m")
    print(f"Final velocity: {velocity[-1]:.3f} m/s")

if __name__ == "__main__":
    main()