# EJ7 - three masses and one pulley

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m1, m2 = 2, 1
k1, k2, k3 = 1, 2, 1
c1, c2, c3 = 0.5, 0.3, 0.5

k23 = k2 * k3 / (k2 + k3)
c23 = c2 * c3 / (c2 + c3)

def f(t):
    return 0.005 * np.cos(0.5 * t)

def fun(t, Y0):
    x1, x1p, y2, y2p = Y0

    x1pp = (1 / m1) * (-(c1 + c23) * x1p - (k1 + k23) * x1 + c23 * y2p + k23 * y2)
    y2pp = (1 / m2) * (f(t) - c23 * y2p - k23 * y2 + c23 * x1p + k23 * x1)

    return [x1p, x1pp, y2p, y2pp]

t_eval = np.arange(0, 50, 0.01)
sol = solve_ivp(fun, [0, 50], [0.1, 0, 0.1, 0], t_eval=t_eval)

print('Solución numérica del sistema:')
print('t[s]        x1           x1p          y2           y2p')
for i in range(0, len(sol.t), 200):
    print(f'{sol.t[i]:8.2f}   {sol.y[0, i]:11.6f}   {sol.y[1, i]:11.6f}   {sol.y[2, i]:11.6f}   {sol.y[3, i]:11.6f}')
print('\nEstado final en t = 50 s:')
print(f'x1 = {sol.y[0, -1]:.6f}')
print(f'x1p = {sol.y[1, -1]:.6f}')
print(f'y2 = {sol.y[2, -1]:.6f}')
print(f'y2p = {sol.y[3, -1]:.6f}')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle('Sistema reducido con polea (Ej. 7a)')

ax1.plot(sol.t, sol.y[0], label='x1')
ax1.plot(sol.t, sol.y[2], label='y2')
ax1.set_ylabel('posición [m]')
ax1.legend()

ax2.plot(sol.t, sol.y[1], label='x1p')
ax2.plot(sol.t, sol.y[3], label='y2p')
ax2.set_ylabel('velocidad [m/s]')
ax2.set_xlabel('t [s]')
ax2.legend()

plt.show()