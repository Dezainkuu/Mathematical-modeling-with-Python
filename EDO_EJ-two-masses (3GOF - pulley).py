import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

m1, m2, J, r = 2, 1, 0.02, 0.1
k1, k2, k3 = 1, 2, 1
c1, c2, c3 = 0.5, 0.3, 0.5

def f(t):
    return 0.005*np.cos(0.5*t)

def sistema(t, X):
    x1, x1p, th, thp, y2, y2p = X
    x1pp = (1/m1)*(-(c1+c2)*x1p + c2*r*thp - (k1+k2)*x1 + k2*r*th)
    thpp = (1/J)*(-r**2*(c2+c3)*thp + r*c2*x1p + r*c3*y2p
                   - r**2*(k2+k3)*th + r*k2*x1 + r*k3*y2)
    y2pp = (1/m2)*(f(t) - c3*y2p + c3*r*thp - k3*y2 + k3*r*th)
    return [x1p, x1pp, thp, thpp, y2p, y2pp]

t_eval = np.arange(0, 50, 0.01)
X0 = [0.1, 0, 0, 0, 0.1, 0]   # x1(0)=0.1, y2(0)=0.1, resto en 0
sol = solve_ivp(sistema, [0, 50], X0, t_eval=t_eval)

x1 = sol.y[0]; rtheta = r*sol.y[2]; y2 = sol.y[4]

plt.plot(sol.t, x1, label='x1')
plt.plot(sol.t, rtheta, label='rθ')
plt.plot(sol.t, y2, label='y2')
plt.xlabel('t [s]'); plt.ylabel('posición [m]'); plt.legend()
plt.title('Sistema con inercia de polea (3 GDL)')
plt.show()