import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#from scipy.integrate import odeint

m1, m2 = 1, 2
k1, k2 = 3, 1
c1, c2 = 0.5, 0.2
"""
def fun(Y0, t):
    y1, y1p, y2, y2p = Y0
    y1pp = (1/m1)*(-(c1+c2)*y1p - (k1+k2)*y1 + c2*y2p + k2*y2)
    y2pp = (1/m2)*(-c2*y2p - k2*y2 + c2*y1p + k2*y1)
    return [y1p, y1pp, y2p, y2pp]

t = np.arange(0, 50, 0.01)
sol = odeint(fun, [0.1, 0, 0, 0], t)   # y1(0)=0.1, resto en 0

plt.plot(t, sol[:,0], label='y1'); plt.plot(t, sol[:,2], label='y2')
plt.legend(); plt.xlabel('t [s]')

# Frecuencias naturales (autovalores de K*M^-1) — cuenta aparte, no depende de odeint
K = np.array([[k1+k2, -k2],[-k2, k2]])
M = np.array([[m1,0],[0,m2]])
w = np.sqrt(np.linalg.eigvals(K @ np.linalg.inv(M)))
f = w/(2*np.pi)
print('Frecuencias naturales:', f, 'Hz')
"""

def fun(t, Y0):                  # <- mismo cambio de orden
    y1, y1p, y2, y2p = Y0
    y1pp = (1/m1)*(-(c1+c2)*y1p - (k1+k2)*y1 + c2*y2p + k2*y2)
    y2pp = (1/m2)*(-c2*y2p - k2*y2 + c2*y1p + k2*y1)
    return [y1p, y1pp, y2p, y2pp]

t_eval = np.arange(0, 50, 0.01)
sol = solve_ivp(fun, [0, 50], [0.1, 0, 0, 0], t_eval=t_eval)

y1 = sol.y[0]
y2 = sol.y[2]
plt.plot(sol.t, y1, label='y1')
plt.plot(sol.t, y2, label='y2')