import numpy as np
from scipy.integrate import odeint

m1, m2 = 2, 1
k1, k2, k3 = 1, 2, 1
c1, c2, c3 = 0.5, 0.3, 0.5

k23 = k2*k3/(k2+k3)
c23 = c2*c3/(c2+c3)

def f(t):
    return 0.005*np.cos(0.5*t)

def fun(Y0, t):
    x1, x1p, y2, y2p = Y0
    x1pp = (1/m1)*(-(c1+c23)*x1p - (k1+k23)*x1 + c23*y2p + k23*y2)
    y2pp = (1/m2)*(f(t) - c23*y2p - k23*y2 + c23*x1p + k23*x1)
    return [x1p, x1pp, y2p, y2pp]

t = np.arange(0, 50, 0.01)
sol = odeint(fun, [0.1, 0, 0.1, 0], t)   # x1(0)=0.1, y2(0)=0.1, resto en 0