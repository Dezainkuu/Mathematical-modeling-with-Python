"""
Sistema de 2 GDL: barra rigida (m, J) pivotada en su centro O, con resortes-
amortiguadores directos a tierra en los puntos A (izquierda, k1,c1) y B
(derecha, k2,c2). Sin masas colgantes (version simple).
 
GDL: y (traslacion del pivote/CM), theta (rotacion). Convencion: +y hacia
arriba, +theta horario (ver DCL). Estado: X = [y, y', theta, theta']
 
Puntos de la barra: yA = y + L1*theta ; yB = y - L2*theta
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# ===========================================================================
# 1) PARAMETROS 
# ===========================================================================
params = dict(
    m=5.0, J=1.5,
    k1=200.0, c1=8.0,
    k2=150.0, c2=6.0,
    L1=0.4, L2=0.6,
)

def f_ext(t):
    """Fuerza/torque externo (0 en este caso base; agregar si el enunciado lo pide)."""
    return 0.0
 
# ===========================================================================
# 2) FUERZAS EN A Y B (checklist propio-vecino; aca el "vecino" es tierra=0)
# ===========================================================================
def fuerzas(y, yp, theta, thetap, p):
    yA  = y  + p['L1']*theta
    yAp = yp + p['L1']*thetap
    yB  = y  - p['L2']*theta
    yBp = yp - p['L2']*thetap
 
    F1 = p['k1']*yA + p['c1']*yAp   # rama en A, a tierra
    F2 = p['k2']*yB + p['c2']*yBp   # rama en B, a tierra
    return F1, F2

# ===========================================================================
# 3) NEWTON: traslacion del CM y rotacion respecto al pivote
# ===========================================================================
def sistema(t, X, p):
    y, yp, theta, thetap = X
    F1, F2 = fuerzas(y, yp, theta, thetap, p)
 
    ypp     = (1/p['m']) * (-F1 - F2)
    thetapp = (1/p['J']) * (-F1*p['L1'] + F2*p['L2'])
    return [yp, ypp, thetap, thetapp]

# ===========================================================================
# 4) SIMULACION
# ===========================================================================
X0 = [0.05, 0, 0.05, 0]   # y(0)=0.05 m, theta(0)=0.05 rad, resto en 0
tmax, dt = 20, 0.01
t_eval = np.linspace(0, tmax, int(tmax/dt) + 1)
 
sol = solve_ivp(sistema, [0, tmax], X0, args=(params,), t_eval=t_eval, method='RK45')
t = sol.t
y, yp, theta, thetap = sol.y
 
# ===========================================================================
# 5) GRAFICOS: posicion/rotacion y velocidad de cada GDL
# ===========================================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
fig.suptitle('Barra 2 GDL — posicion/rotacion y velocidad')
 
axs[0,0].plot(t, y);       axs[0,0].set_ylabel('y [m]');          axs[0,0].set_title('Posicion / Rotacion')
axs[0,1].plot(t, yp, 'tab:orange');     axs[0,1].set_ylabel("y' [m/s]"); axs[0,1].set_title('Velocidad')
axs[1,0].plot(t, theta, 'tab:green');   axs[1,0].set_ylabel('theta [rad]'); axs[1,0].set_xlabel('t [s]')
axs[1,1].plot(t, thetap, 'tab:red');    axs[1,1].set_ylabel('thetap [rad/s]'); axs[1,1].set_xlabel('t [s]')
plt.tight_layout()
plt.show()