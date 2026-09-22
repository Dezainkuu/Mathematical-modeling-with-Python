"""
Ejercicio 7, punto g) - Sistema completo (3 GDL, J=0.02 Kg*m^2) excitado con
una fuerza armonica cuya frecuencia coincide con la primera frecuencia natural
hallada en d), amplitud 0.005 N.

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


def modos_propios(p):
    """Frecuencias naturales [Hz y rad/s] y formas modales del sistema."""
    M, K, _ = matrices_MKC(p)
    vals, vecs = np.linalg.eig(K @ np.linalg.inv(M))
    idx = np.argsort(vals)
    w = np.sqrt(vals[idx])
    return w, w/(2*np.pi), vecs[:, idx], M


# ===========================================================================
# SIMULACION
# ===========================================================================
p = dict(params_base, J=0.02)

# Primera frecuencia natural (exacta, sin redondear) del sistema con inercia
w, f_hz, _, _ = modos_propios(p)
omega1 = w[0]
print(f"Primera frecuencia natural: f1 = {f_hz[0]:.4f} Hz  (omega1 = {omega1:.4f} rad/s)")

def f_resonante(t):
    return 0.005*np.cos(omega1*t)

X0 = [0.1, 0, 0, 0, 0.1, 0]
tmax, dt = 50, 0.01
t_eval = np.arange(0, tmax + dt, dt)

sol = solve_ivp(sistema, [0, tmax], X0, args=(p, f_resonante),
                 t_eval=t_eval, method='RK45')

# ===========================================================================
# GRAFICOS
# ===========================================================================
fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
fig.suptitle(f'Ejercicio 7g) - Excitacion en resonancia (f = f1 = {f_hz[0]:.4f} Hz)')

r = p['r']
etiquetas = [r'$x_1$ [m]', r'$r\theta$ [m]', r'$y_2$ [m]']
indices = [0, 2, 4]

for ax, idx, lab in zip(axs, indices, etiquetas):
    escala = r if idx == 2 else 1.0
    ax.plot(sol.t, sol.y[idx]*escala, color='tab:red')
    ax.set_ylabel(lab)

axs[-1].set_xlabel('t [s]')
plt.tight_layout()
plt.show()