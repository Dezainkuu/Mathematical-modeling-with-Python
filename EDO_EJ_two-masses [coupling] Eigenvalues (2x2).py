import numpy as np

m1, m2 = 2, 1
k1, k2, k3 = 1, 2, 1
k23 = k2*k3/(k2+k3)

K = np.array([[k1+k23, -k23], [-k23, k23]])
M = np.array([[m1, 0], [0, m2]])

vals, vecs = np.linalg.eig(K @ np.linalg.inv(M))
idx = np.argsort(vals)
w = np.sqrt(vals[idx]); f = w/(2*np.pi)
vecs = vecs[:, idx]

print("Modo | Frecuencia [Hz] | x1 | y2")
for i in range(2):
    v = vecs[:,i]/np.max(np.abs(vecs[:,i]))
    print(f"{i+1}    | {f[i]:.4f}          | {v[0]:.3f} | {v[1]:.3f}")