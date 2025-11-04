import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
z = np.sin(np.sqrt(X**2 + Y**2))

plt.contour(X, Y, z, levels=20, cmap='viridis')
plt.colorbar(label='z value')
plt.title('Contour plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()