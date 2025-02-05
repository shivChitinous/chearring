import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def kuramotoLEDs(n=42, omega=0.1, K=0.1, time_steps=200, heterogeneity=0.02):
    """
    Simulates a 1D array of n coupled oscillators using the Kuramoto model.
    Returns the amplitude (cosine of phase) over time.

    Parameters:
    - n: Number of oscillators.
    - omega: Base natural frequency.
    - K: Coupling strength.
    - time_steps: Number of time steps.
    - heterogeneity: Standard deviation for random natural frequencies.

    Returns:
    - A NumPy array of shape (time_steps, n) containing amplitude values over time.
    """

    # Initialize random phases in [0, 2π]
    theta = np.random.uniform(0, 2*np.pi, n)
    
    # Assign each oscillator a slightly different natural frequency
    omega_n = omega + heterogeneity * np.random.randn(n)

    # Store amplitude evolution over time
    amplitude = np.zeros((time_steps, n))

    # Simulate over time
    for t in range(time_steps):
        theta_new = np.copy(theta)

        # Compute global coupling term for each oscillator
        global_coupling = np.sum(np.sin(theta[:, np.newaxis] - theta), axis=0) / n

        # Update phase using the Kuramoto equation
        theta_new += omega_n + K * global_coupling

        # Apply modulo 2π to keep phases within range
        theta = np.mod(theta_new, 2*np.pi)

        # Store the amplitude (cosine of phase)
        amplitude[t] = np.cos(theta)

    return amplitude  # Shape: (time_steps, n)

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((self.size, self.size))
    
    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1).copy()
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        self.weights /= len(patterns)  # Normalize weights
    
    def recall(self, pattern, steps=500, updateNeurons = 10, noise=0.1, pushFrequency = 100, pushPattern = None):
        initPattern = pattern.copy()
        recalled_patterns = []
        for _ in range(steps):
            indices = np.arange(self.size)
            np.random.shuffle(indices)  # Asynchronous update order
            for i in indices:
                pattern[i] = 1 if np.dot(self.weights[i], pattern) >= 0 else -1
                if i % updateNeurons == 0:
                    pattern += noise * np.random.choice([-1, 1], size=len(pattern))  # Add noise
                    recalled_patterns.append(pattern.copy())  # Save after nth neuron update
            if _ % pushFrequency == 0:
                if pushPattern is not None:
                    pattern = pushPattern.copy()
                else:
                    pattern = initPattern.copy()  # Push to limit
        return np.array(recalled_patterns)

#board image of owlet
def owlet(status = 'sleeping', size = 42):
    activations = -np.ones(size)
    activations[[23, 12, 11, 27, 28, 39, 0]] = 1
    if status=='not sleeping':
        activations[[4, 1, 6, 34, 36, 40, 29, 13]] = 1

    elif status=='sleeping':
        activations[[5, 35]] = 1
    elif status == 'wink':
        activations[[5, 34, 36, 40, 29]] = 1
    return activations