import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create grid of x, y values
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Define the well function
A = 0.1
sigma = 0.5
n = 100
u = 10*np.random.normal(0,0.25,(2,1,1,n))
print(u[0].shape)
Z = -A * np.exp(-((np.expand_dims(X, -1) - u[0])**2 + (np.expand_dims(Y, -1) - u[1])**2) / (2 * sigma**2))
Z = np.sum(Z, axis=-1)
#Z = (-1/ np.sqrt((X**2+Y**2)/A))

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plane with Wells')
plt.show()
