import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================
# Sistema de 2 GDL horizontal: m1-k1,c1-pared, m1-k12,c12-m2, m2-k2,c2-pared
# Variables de estado (mismas que en el planteo a mano):
#   y1 = x1 , y2 = x1_dot , y3 = x2 , y4 = x2_dot
# ============================================================

# ---------- Parametros del sistema (reemplazar por los del enunciado) ----------
m1, m2 = 1.0, 1.5                     # kg
k1, k12, k2 = 200.0, 150.0, 180.0     # N/m
c1, c12, c2 = 2.0, 1.5, 2.5           # N*s/m


def sistema(t, y):
    y1, y2, y3, y4 = y
    dy1 = y2
    dy3 = y4
    dy2 = (1/m1) * (k12*y3 + c12*y4 - k1*y1 - c1*y2 - k12*y1 - c12*y2)
    dy4 = (1/m2) * (k12*y1 + c12*y2 - k12*y3 - c12*y4 - k2*y3 - c2*y4)
    return [dy1, dy2, dy3, dy4]


# ---------- Condiciones iniciales: se aparta m1 y se suelta el sistema ----------
y0 = [0.05, 0.0, 0.0, 0.0]     # x1(0)=0.05 m, resto en reposo

t_span = (0.0, 5.0)
t_eval = np.linspace(*t_span, 2000)

sol = solve_ivp(sistema, t_span, y0, t_eval=t_eval, method="RK45")
x1, x2 = sol.y[0], sol.y[2]

# ---------- Grafico de la respuesta temporal ----------
plt.figure(figsize=(9, 4.5))
plt.plot(sol.t, x1, label="x1(t)")
plt.plot(sol.t, x2, label="x2(t)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Desplazamiento [m]")
plt.title("Respuesta libre del sistema de 2 GDL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/respuesta_2gdl.png", dpi=150)

# ---------- Extra: frecuencias naturales del sistema no amortiguado ----------
M = np.array([[m1, 0],
              [0, m2]])
K = np.array([[k1 + k12, -k12],
              [-k12,      k12 + k2]])

autovalores = np.linalg.eigvals(np.linalg.inv(M) @ K)
omegas = np.sqrt(np.sort(autovalores.real))

print("Frecuencias naturales [rad/s]:", omegas)
print("Frecuencias naturales [Hz]   :", omegas / (2*np.pi))
print("\nx1(0.5s) =", x1[np.argmin(np.abs(sol.t-0.5))])
print("x2(0.5s) =", x2[np.argmin(np.abs(sol.t-0.5))])