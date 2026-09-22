"""
Ejercicio 7, punto f) - Sistema completo (3 GDL, J=0.02 Kg*m^2), excitado con
un escalon: f(t)=0 para t<10s, f(t)=0.005N (constante) para t>=10s.
Compara la respuesta transitoria con la obtenida en el punto c) (excitacion
armonica f(t)=0.005*cos(0.5t)).

Script autocontenido (no depende de otros archivos).
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===========================================================================
# PARAMETROS DEL SISTEMA (Ejercicio 7 - polea con inercia J)
# ===========================================================================
params_base = dict(
    m1=2.0, m2=1.0,
    k1=1.0, k2=2.0, k3=1.0,
    c1=0.5, c2=0.3, c3=0.5,
    r=0.1,
    J=0.02,   # valor nominal (puntos c-d)
)


def matrices_MKC(p):
    """Matrices de masa, rigidez y amortiguamiento (orden: x1, theta, y2)."""
    m1, m2, J, r = p['m1'], p['m2'], p['J'], p['r']
    k1, k2, k3 = p['k1'], p['k2'], p['k3']
    c1, c2, c3 = p['c1'], p['c2'], p['c3']

    M = np.diag([m1, J, m2])
    K = np.array([
        [k1+k2,   -k2*r,             0    ],
        [-k2*r,    r**2*(k2+k3),    -k3*r ],
        [0,       -k3*r,             k3   ],
    ])
    C = np.array([
        [c1+c2,   -c2*r,             0    ],
        [-c2*r,    r**2*(c2+c3),    -c3*r ],
        [0,       -c3*r,             c3   ],
    ])
    return M, K, C


def sistema(t, X, p, f_func):
    """ODE del sistema completo (3 GDL). f_func(t) es la fuerza sobre m2."""
    q  = np.array([X[0], X[2], X[4]])
    qp = np.array([X[1], X[3], X[5]])

    M, K, C = matrices_MKC(p)
    F = np.array([0.0, 0.0, f_func(t)])

    qpp = np.linalg.solve(M, F - C @ qp - K @ q)
    return [qp[0], qpp[0], qp[1], qpp[1], qp[2], qpp[2]]


# ===========================================================================
# SIMULACION
# ===========================================================================
p = dict(params_base, J=0.02)   # J nominal

X0 = [0.1, 0, 0, 0, 0.1, 0]     # x1(0)=0.1, theta(0)=0, y2(0)=0.1 (dato original)
tmax, dt = 50, 0.01
t_eval = np.arange(0, tmax + dt, dt)

# --- Excitacion escalon (punto f) ---
def f_escalon(t):
    return 0.0 if t < 10 else 0.005

sol_escalon = solve_ivp(sistema, [0, tmax], X0, args=(p, f_escalon),
                         t_eval=t_eval, method='RK45')

# --- Excitacion armonica (punto c, para comparar) ---
def f_armonica(t):
    return 0.005*np.cos(0.5*t)

sol_armonica = solve_ivp(sistema, [0, tmax], X0, args=(p, f_armonica),
                          t_eval=t_eval, method='RK45')

# ===========================================================================
# GRAFICOS COMPARATIVOS
# ===========================================================================
fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
fig.suptitle('Ejercicio 7f) - Respuesta a escalon vs. armonica (J=0.02 Kg$\\cdot$m$^2$)')

etiquetas = [r'$x_1$ [m]', r'$r\theta$ [m]', r'$y_2$ [m]']
indices = [0, 2, 4]   # posiciones en el vector de estado
r = p['r']

for ax, idx, lab in zip(axs, indices, etiquetas):
    escala = r if idx == 2 else 1.0
    ax.plot(sol_escalon.t, sol_escalon.y[idx]*escala, label='Escalon (f)', color='tab:blue')
    ax.plot(sol_armonica.t, sol_armonica.y[idx]*escala, label='Armonica (c)',
            color='tab:orange', alpha=0.7)
    ax.set_ylabel(lab)
    ax.legend()

axs[-1].set_xlabel('t [s]')
plt.tight_layout()
plt.show()