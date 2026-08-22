import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate


def f1(x, y):
    return (np.sqrt(25+x**2+y**2))/(np.sin(1+x**2+y**2)*np.cos(1+x+y))

def f2(x, y):
    return (np.sqrt(((x-1)**2)*((y-1)**2)))

def f3(x, y):
    return (x+2)/(x**2+y**2)

x = np.linspace(-10,10,100)
y = np.linspace(-10,10,100)

X, Y = np.meshgrid(x,y)
Z = f3(X,Y)

plt.pcolor(X, Y, Z, cmap=plt.cm.jet,edgecolors='k', linewidth=0.1)

# Plot the surface.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(15,10))

# Customize the z axis.
ax.set_zlim(-100, 100)

surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.jet,edgecolors='k',
                       linewidth=0.1, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()