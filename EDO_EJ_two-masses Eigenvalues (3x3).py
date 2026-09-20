import numpy as np

m1, m2, J, r = 2, 1, 0.02, 0.1
k1, k2, k3 = 1, 2, 1

K = np.array([[k1+k2, -k2*r, 0],
              [-k2*r, r**2*(k2+k3), -k3*r],
              [0, -k3*r, k3]])
M = np.diag([m1, J, m2])

vals, vecs = np.linalg.eig(K @ np.linalg.inv(M))
idx = np.argsort(vals)
w = np.sqrt(vals[idx]); f = w/(2*np.pi)
vecs = vecs[:, idx]

print("Modo | Frecuencia [Hz] | x1 | r*theta | y2")
for i in range(3):
    phi = np.linalg.inv(M) @ vecs[:,i]
    phi[1] *= r                                # convierte theta a r*theta
    phi = phi/np.max(np.abs(phi))
    print(f"{i+1}    | {f[i]:.4f}          | {phi[0]:.3f} | {phi[1]:.3f}    | {phi[2]:.3f}")