"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

m, k1, k2, c1, c2 = 1, 1, 3, 0.5, 0.3
w_f = 0.2   # frecuencia de la fuerza, cambia a 2 en la parte d)

def f(t):
    return 0.05*np.cos(w_f*t)

def sistema(X0, t):
    x1, x2 = X0
    x1p = x2
    x2p = (1/m)*(f(t) - (c1+c2)*x2 - (k1+k2)*x1)
    return [x1p, x2p]

t = np.arange(0, 50, 0.01)
sol = odeint(sistema, [0.1, 0], t)   # x0 = 0.1, v0 = 0
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m, k1, k2, c1, c2 = 1, 1, 3, 0.5, 0.3
w_f = 0.2

def f(t):
    return 0.05*np.cos(w_f*t)

def sistema(t, X0): 
    x1, x2 = X0
    x1p = x2
    x2p = (1/m)*(f(t) - (c1+c2)*x2 - (k1+k2)*x1)
    return [x1p, x2p]

t_eval = np.arange(0, 50, 0.01)
sol = solve_ivp(sistema, [0, 50], [0.1, 0], t_eval=t_eval)

x = sol.y[0]      # posición
v = sol.y[1]      # velocidad
plt.plot(sol.t, x)