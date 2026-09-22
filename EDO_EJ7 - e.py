"""
Ejercicio 7, punto e) - Frecuencias naturales en funcion de la inercia J
de la polea. Compara con J=0.0002, J=0.02 (nominal) y J=0.2 Kg*m^2, y grafica
un barrido logaritmico de J junto con las frecuencias del modelo reducido.

Script autocontenido (no depende de otros archivos).
"""
import numpy as np
import matplotlib.pyplot as plt

# ===========================================================================
# PARAMETROS DEL SISTEMA (Ejercicio 7 - polea con inercia J)
# ===========================================================================
params_base = dict(
    m1=2.0, m2=1.0,
    k1=1.0, k2=2.0, k3=1.0,
    c1=0.5, c2=0.3, c3=0.5,
    r=0.1,
    J=0.02,   # valor nominal (puntos c-d); se sobreescribe mas abajo
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


def modos_propios(p):
    """Frecuencias naturales [Hz] y formas modales del sistema con inercia."""
    M, K, _ = matrices_MKC(p)
    vals, vecs = np.linalg.eig(K @ np.linalg.inv(M))
    idx = np.argsort(vals)
    w = np.sqrt(vals[idx])
    return w, w/(2*np.pi), vecs[:, idx], M


def frecuencias_reducidas(p):
    """Frecuencias naturales del sistema reducido de 2 GDL (punto b), J->0."""
    k2, k3 = p['k2'], p['k3']
    k23 = k2*k3/(k2+k3)
    K = np.array([[p['k1']+k23, -k23], [-k23, k23]])
    M = np.diag([p['m1'], p['m2']])
    vals = np.linalg.eigvals(K @ np.linalg.inv(M))
    return np.sort(np.sqrt(vals)/(2*np.pi))


# ===========================================================================
# VALORES PUNTUALES PEDIDOS
# ===========================================================================
print("J [Kg*m^2] |  f1 [Hz]  |  f2 [Hz]  |  f3 [Hz]")
for J in [0.0002, 0.02, 0.2]:
    p = dict(params_base, J=J)
    _, f_hz, _, _ = modos_propios(p)
    print(f"{J:>10} | {f_hz[0]:.4f}   | {f_hz[1]:.4f}   | {f_hz[2]:.4f}")

f_reducido = frecuencias_reducidas(params_base)
print(f"\nSistema reducido (b): f1={f_reducido[0]:.4f} Hz, f2={f_reducido[1]:.4f} Hz")

# ===========================================================================
# BARRIDO LOGARITMICO DE J
# ===========================================================================
J_vals = np.logspace(-5, 2, 300)
f1s, f2s, f3s = [], [], []
for J in J_vals:
    p = dict(params_base, J=J)
    _, f_hz, _, _ = modos_propios(p)
    f1s.append(f_hz[0]); f2s.append(f_hz[1]); f3s.append(f_hz[2])

plt.figure(figsize=(8, 5))
plt.loglog(J_vals, f1s, label='$f_1$ (3 GDL)')
plt.loglog(J_vals, f2s, label='$f_2$ (3 GDL)')
plt.loglog(J_vals, f3s, label='$f_3$ (3 GDL)')
plt.axhline(f_reducido[0], color='gray', ls='--', label='$f_1$ reducido (b)')
plt.axhline(f_reducido[1], color='black', ls='--', label='$f_2$ reducido (b)')
plt.axvline(0.02, color='red', ls=':', label='J nominal (0.02)')
plt.xlabel('J [Kg$\\cdot$m$^2$] (escala log)')
plt.ylabel('Frecuencia natural [Hz] (escala log)')
plt.title('Frecuencias naturales vs. inercia de la polea')
plt.legend()
plt.tight_layout()
plt.show()

# ===========================================================================
# LIMITE J MUY GRANDE (verificacion fisica)
# ===========================================================================
print("\nLimite J grande (verificacion):")
for J in [2, 20, 200]:
    p = dict(params_base, J=J)
    _, f_hz, _, _ = modos_propios(p)
    print(f"  J={J}: f1={f_hz[0]:.4f} Hz, f2={f_hz[1]:.4f} Hz, f3={f_hz[2]:.4f} Hz")

f_m1_sola = np.sqrt((params_base['k1']+params_base['k2'])/params_base['m1'])/(2*np.pi)
f_m2_sola = np.sqrt(params_base['k3']/params_base['m2'])/(2*np.pi)
print(f"\nm1 sola contra k1+k2 (polea fija): {f_m1_sola:.4f} Hz")
print(f"m2 sola contra k3 (polea fija): {f_m2_sola:.4f} Hz")