"""
Ejercicio 9 - Disco (J1) con dos radios de arrollamiento (r1, r2), conectado
por cable inextensible a: una pared via k1,c1 (radio r1), y a una masa m
via k2,c2 (radio r2). La masa m tambien esta conectada a una base movil
y_e(t) via un resorte k3 (sin amortiguador, segun la Figura 4).

Grados de libertad:
    y      -> posicion de la masa m
    theta  -> rotacion del disco

Convencion de signos: +y hacia abajo, +theta antihorario (segun el DCL
de la catedra). Vector de estado: X = [y, y', theta, theta']

Cinematica (cable inextensible, ambos cables se enrollan a favor de theta):
    y_A = r1*theta   (punto del disco hacia la pared, rama k1,c1)
    y_B = r2*theta   (punto del disco hacia m, rama k2,c2)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ===========================================================================
# 1) PARAMETROS DEL SISTEMA -> cambiar aca para reutilizar con otros datos
# ===========================================================================
params = dict(
    k1=2.0, c1=0.2,      # rama disco (radio r1) <-> pared
    k2=1.0, c2=0.3,      # rama disco (radio r2) <-> masa m
    k3=1.0,               # rama masa m <-> base y_e(t)  (SIN amortiguador, ver figura)
    m=1.0,                 # masa [kg]
    J1=0.3,                # inercia rotacional del disco [kg*m^2]
    r1=0.3, r2=0.5,         # radios de arrollamiento [m]
)


# ===========================================================================
# 2) EXCITACION CONOCIDA: y_e(t)
# ===========================================================================
def ye(t):
    return 0.01*np.sin(t)


# ===========================================================================
# 3) FUERZAS DE CADA RAMA, segun el DCL (convencion de la catedra: F1 y F2
#    armadas desde la "optica" del disco; F3 desde la optica de m)
# ===========================================================================
def fuerzas(t, y, yp, theta, thetap, p):
    r1, r2 = p['r1'], p['r2']

    F1 = p['k1']*r1*theta + p['c1']*r1*thetap                       # rama disco <-> pared
    F2 = p['k2']*(r2*theta - y) + p['c2']*(r2*thetap - yp)          # rama disco <-> m
    F3 = p['k3']*(y - ye(t))                                        # rama m <-> y_e (sin amortiguador)

    return F1, F2, F3


# ===========================================================================
# 4) FUNCION ODE: Newton (traslacion) en m ; Newton rotacional en el disco
#    m*y''     = F2 - F3           (F2 ya viene "desde el disco": su reaccion
#                                    sobre m es -F2, y -(-F2) = F2 por el doble
#                                    signo ya explicado)
#    J1*theta''= -r1*F1 - r2*F2    (F1, F2 escritas desde el disco -> ambas
#                                    restan en la ecuacion propia del disco)
# ===========================================================================
def sistema(t, X, p):
    y, yp, theta, thetap = X

    F1, F2, F3 = fuerzas(t, y, yp, theta, thetap, p)

    ypp     = (1/p['m'])  * (F2 - F3)
    thetapp = (1/p['J1']) * (-p['r1']*F1 - p['r2']*F2)

    return [yp, ypp, thetap, thetapp]


# ===========================================================================
# 5) CONDICIONES INICIALES Y SIMULACION
# ===========================================================================
# Orden: [y, y', theta, theta']
X0 = [0, 0, 0.1, 0]     # angulo inicial del disco = 0.1 rad (dato)

tmax, dt = 100, 0.01
t_eval = np.arange(0, tmax + dt, dt)

sol = solve_ivp(sistema, [0, tmax], X0, args=(params,), t_eval=t_eval, method='RK45')
t = sol.t
y, yp, theta, thetap = sol.y


# ===========================================================================
# 6) GRAFICOS: posicion/rotacion y velocidad/velocidad angular
# ===========================================================================
fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
fig.suptitle('Ejercicio 9 - Posicion/rotacion y velocidad de cada GDL')

axs[0, 0].plot(t, y)
axs[0, 0].set_ylabel('y [m]')
axs[0, 0].set_title('Posicion / Rotacion')

axs[0, 1].plot(t, yp, color='tab:orange')
axs[0, 1].set_ylabel("y' [m/s]")
axs[0, 1].set_title('Velocidad / Vel. angular')

axs[1, 0].plot(t, theta, color='tab:green')
axs[1, 0].set_ylabel(r'$\theta$ [rad]')
axs[1, 0].set_xlabel('t [s]')

axs[1, 1].plot(t, thetap, color='tab:red')
axs[1, 1].set_ylabel(r'$\dot\theta$ [rad/s]')
axs[1, 1].set_xlabel('t [s]')

plt.tight_layout()
plt.show()


# ===========================================================================
# 7) MODOS PROPIOS: matrices M, K (sin amortiguamiento ni fuerzas externas)
#    Orden de coordenadas: [y, theta]  (mismo orden que las posiciones del estado)
# ===========================================================================
def matrices_MK(p):
    k1, k2, k3 = p['k1'], p['k2'], p['k3']
    m, J1 = p['m'], p['J1']
    r1, r2 = p['r1'], p['r2']

    M = np.diag([m, J1])

    K = np.array([
        [k2+k3,        -k2*r2],
        [-k2*r2,        k1*r1**2 + k2*r2**2],
    ])
    return M, K

M, K = matrices_MK(params)

vals, vecs = np.linalg.eig(K @ np.linalg.inv(M))
idx = np.argsort(vals)
w = np.sqrt(vals[idx])
f_hz = w/(2*np.pi)
vecs = vecs[:, idx]

print("Frecuencias naturales:")
for i in range(2):
    print(f"  Modo {i+1}: omega = {w[i]:.4f} rad/s  ({f_hz[i]:.4f} Hz)")

print("\nFormas modales normalizadas (y, theta):")
for i in range(2):
    phi = np.linalg.inv(M) @ vecs[:, i]
    phi = phi/np.max(np.abs(phi))
    print(f"  Modo {i+1}: y={phi[0]:.3f}  theta={phi[1]:.3f}")