import numpy as np

m1, m2 = 2, 1
k1, k2, k3 = 1, 2, 1
k23 = k2*k3/(k2+k3)

K = np.array([[k1+k23, -k23], [-k23, k23]])
M = np.array([[m1, 0], [0, m2]])

w = np.sqrt(np.linalg.eigvals(K @ np.linalg.inv(M)))
f = w/(2*np.pi)
print('Frecuencias naturales:', f, 'Hz')