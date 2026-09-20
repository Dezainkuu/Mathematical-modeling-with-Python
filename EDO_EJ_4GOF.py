"""
Ejercicio 8 - Barra rigida (m1, J) con dos masas colgantes (m2, m3),
excitadas en la base (yp1, yp2), y fuerza externa Fe(t) aplicada en el
punto C de la barra (a distancia L3 del centro de masa).
 
Grados de libertad (orden usado en todo el codigo):
    y2, y3  -> posicion de las masas colgantes
    y1      -> traslacion del centro de masa (CM) de la barra
    theta   -> rotacion de la barra
 
Convencion de signos: +y hacia abajo, +theta antihorario.
Puntos de la barra: A (izquierda, a distancia L1+L3 del CM),
                     B (derecha, a distancia L2 del CM),
                     C (donde se aplica Fe, a distancia L3 del CM).
"""
 
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
 
# ===========================================================================
# 1) PARAMETROS DEL SISTEMA 
# ===========================================================================
params = dict(
    k1=2.0, k2=1.0, k3=1.0, k4=2.0,      # rigideces [N/m]
    c1=0.1, c2=0.5, c3=0.5, c4=0.1,      # amortiguamientos [Ns/m]
    m1=5.0, m2=1.0, m3=1.0,              # masas [kg]  (m1 = masa de la barra)
    J=0.3,                                # inercia rotacional de la barra [kg*m^2]
    L1=0.25, L2=0.5, L3=0.25,             # distancias [m] (ver Figura 3)
)
 
def brazos(p):
    """Brazos de palanca de los puntos A y B respecto al centro de masa (pivote)."""
    brazo_A = p['L1'] + p['L3']   # A queda del lado izquierdo, a esta distancia del CM
    brazo_B = p['L2']             # B queda del lado derecho, a esta distancia del CM
    return brazo_A, brazo_B
 
 
# ===========================================================================
# 2) EXCITACIONES CONOCIDAS: yp1(t), yp2(t), Fe(t) (y sus derivadas para c*)
# ===========================================================================
def yp1(t):   return 0.001*np.cos(2*t)
def dyp1(t):  return -0.001*2*np.sin(2*t)
 
def yp2(t):   return 0.001*np.cos(2*t + 0.1)
def dyp2(t):  return -0.001*2*np.sin(2*t + 0.1)
 
def Fe(t):    return 0.1*np.cos(2*t)
 
 
# ===========================================================================
# 3) FUERZAS DE CADA RAMA (resorte+amortiguador), segun el DCL
#    F = k*(propio - vecino) + c*(propio' - vecino')
# ===========================================================================
def fuerzas(t, y2, y2p, y3, y3p, y1, y1p, theta, thetap, p):
    brazo_A, brazo_B = brazos(p)
 
    # Puntos A y B de la barra: se mueven por la traslacion Y la rotacion del CM
    yA  = y1  + brazo_A*theta
    yAp = y1p + brazo_A*thetap
    yB  = y1  - brazo_B*theta
    yBp = y1p - brazo_B*thetap
 
    F1 = p['k1']*(y2 - yp1(t)) + p['c1']*(y2p - dyp1(t))   # rama m2 <-> base yp1
    F2 = p['k2']*(y2 - yA)     + p['c2']*(y2p - yAp)       # rama m2 <-> punto A
    F3 = p['k3']*(y3 - yB)     + p['c3']*(y3p - yBp)       # rama m3 <-> punto B
    F4 = p['k4']*(y3 - yp2(t)) + p['c4']*(y3p - dyp2(t))   # rama m3 <-> base yp2
 
    return F1, F2, F3, F4
 
 
# ===========================================================================
# 4) FUNCION ODE: Newton (traslacion) en m2, m3, m1 ; Newton rotacional en J
# ===========================================================================
def sistema(t, X, p):
    y2, y2p, y3, y3p, y1, y1p, theta, thetap = X
    brazo_A, brazo_B = brazos(p)
 
    F1, F2, F3, F4 = fuerzas(t, y2, y2p, y3, y3p, y1, y1p, theta, thetap, p)
 
    y2pp    = (1/p['m2']) * (-F1 - F2)
    y3pp    = (1/p['m3']) * (-F4 - F3)
    y1pp    = (1/p['m1']) * (F2 + F3 + Fe(t))
    thetapp = (1/p['J'])  * (brazo_A*F2 - brazo_B*F3 + p['L3']*Fe(t))
 
    return [y2p, y2pp, y3p, y3pp, y1p, y1pp, thetap, thetapp]
 
 
# ===========================================================================
# 5) CONDICIONES INICIALES Y SIMULACION
# ===========================================================================
# Orden: [y2, y2p, y3, y3p, y1, y1p, theta, thetap]
X0 = [0, 0, 0, 0, 0, 0, 0.1, 0]   # angulo inicial de la barra = 0.1 rad (dato)
 
tmax, dt = 100, 0.01
t_eval = np.arange(0, tmax + dt, dt)
 
sol = solve_ivp(sistema, [0, tmax], X0, args=(params,), t_eval=t_eval, method='RK45')
t = sol.t
y2, y2p, y3, y3p, y1, y1p, theta, thetap = sol.y
 
 
# ===========================================================================
# 6) GRAFICOS: posicion y velocidad de cada grado de libertad
# ===========================================================================
fig, axs = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
fig.suptitle('Ejercicio 8 - Posicion y velocidad de cada GDL')
 
datos = [
    (r'$y_2$',     y2,    y2p,    'm'),
    (r'$y_3$',     y3,    y3p,    'm'),
    (r'$y_1$',     y1,    y1p,    'm'),
    (r'$\theta$',  theta, thetap, 'rad'),
]
 
for i, (nombre, pos, vel, unidad) in enumerate(datos):
    axs[i, 0].plot(t, pos)
    axs[i, 0].set_ylabel(f'{nombre} [{unidad}]')
    axs[i, 1].plot(t, vel, color='tab:orange')
    axs[i, 1].set_ylabel(f'{nombre}p [{unidad}/s]')
 
axs[0, 0].set_title('Posicion')
axs[0, 1].set_title('Velocidad')
axs[-1, 0].set_xlabel('t [s]')
axs[-1, 1].set_xlabel('t [s]')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/ejercicio8_respuesta.png', dpi=130)
print("Grafico guardado.")
 
 
# ===========================================================================
# 7) MODOS PROPIOS: matrices M, K (sin C ni fuerzas) y autovalores/autovectores
#    Orden de coordenadas: [y2, y3, y1, theta]  (mismo orden que el estado)
# ===========================================================================
def matrices_MK(p):
    brazo_A, brazo_B = brazos(p)
    k1, k2, k3, k4 = p['k1'], p['k2'], p['k3'], p['k4']
    m1, m2, m3, J  = p['m1'], p['m2'], p['m3'], p['J']
 
    M = np.diag([m2, m3, m1, J])
 
    K = np.array([
        [k1+k2,        0,          -k2,                  -brazo_A*k2],
        [0,            k3+k4,      -k3,                   brazo_B*k3],
        [-k2,         -k3,          k2+k3,                brazo_A*k2 - brazo_B*k3],
        [-brazo_A*k2,  brazo_B*k3,  brazo_A*k2-brazo_B*k3, brazo_A**2*k2 + brazo_B**2*k3],
    ])
    return M, K
 
M, K = matrices_MK(params)
 
vals, vecs = np.linalg.eig(K @ np.linalg.inv(M))
idx = np.argsort(vals)
w = np.sqrt(vals[idx])
f_hz = w/(2*np.pi)
vecs = vecs[:, idx]
 
print("\nFrecuencias naturales [Hz]:")
for i, f in enumerate(f_hz):
    print(f"  Modo {i+1}: {f:.4f} Hz")
 
print("\nFormas modales normalizadas (y2, y3, y1, theta):")
for i in range(4):
    phi = np.linalg.inv(M) @ vecs[:, i]
    phi = phi/np.max(np.abs(phi))
    print(f"  Modo {i+1} (f={f_hz[i]:.4f} Hz): "
          f"y2={phi[0]:.3f}  y3={phi[1]:.3f}  y1={phi[2]:.3f}  theta={phi[3]:.3f}")