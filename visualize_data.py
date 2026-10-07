import numpy as np
import matplotlib.pyplot as plt

data = np.load('//data/0f3a9a6d-cde0-4582-8cd5-c29ab7e4eb25.npy')

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")

plt.plot(data[:1000])
plt.title("EEG Data - First Second (4 Channels)")
plt.xlabel("Samples")
plt.ylabel("Signal Value")
plt.show()