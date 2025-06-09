import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt



def sensor1(N=1000):
    x = np.random.normal(0,.1,N)

    return x

def sensor2(t):
    x = np.random.normal(0,1,1)[0]

    return x*t**2


def lpf(omega,N,data,dt):

    tau = 1/omega
    alpha = dt/(tau + dt)
    

    filtered_data = np.zeros(len(data))
    
    for i in range(N):
        if i == 0:
            y = data[0]
        else:
            y = alpha * data[i] + (1 - alpha) * y
   

        filtered_data[i] = y

    return filtered_data


def sci_lowpass(omega,data,dt):
    b, a = signal.butter(1, omega , btype='low',fs=1/dt)
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data



def main():
    T = 1
    N = 1000
    dt = T/N
    np.random.seed(0)  # For reproducibility

    time = np.arange(0, T, dt)
    
    omega = 5
    data = sensor1(N)
    y = lpf(omega*1.5,N,data,dt)
    y1 = sci_lowpass(omega,data,dt)



    plt.figure(figsize=(10, 5))
    plt.title('Low Pass Filtered Signal')
    plt.plot(time,y,label='Filtered Signal')
    plt.plot(time,y1,label='SciPy Filtered Signal', linestyle='-')
    plt.plot(time,data,'-.',label='Original Signal', alpha=0.25)
    plt.xlabel('Time (s)')  
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


