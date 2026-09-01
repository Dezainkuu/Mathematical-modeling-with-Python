import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

def fun1(x):
    f = np.cos(4*x) * np.cos (3 * np.sin(x))
    return f

#método de integración por punto medio
def punto_medio (a, b, n):
    h = (b - a)/n  # Ancho de los subintervalos
    x = np.linspace (a+0.5*h, b-0.5*h, n)
    y = fun1 (x)
    I_pm = h*np.sum(y)
    return I_pm

#método de integración trapezoidal
def trapezoidal (a, b, n):
    h = (b - a)/n
    x = np.linspace (a, b, n+1) #n son los puntos, por lo que necesitamos n+1 para los intervalos y el recorrido correcto
    y = fun1 (x)
    S = y[0] + 2*np.sum(y[2:n-1]) + y[n]
    I_trpz = 0.5*h*S
    return I_trpz


#método de integración simpson 1/3
def simpson13(a, b, n): # El método requiere estrictamente un número par de intervalos
    
    if n % 2 != 0:
        raise ValueError("El número de subintervalos 'n' debe ser un número par.")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = fun1(x) # Reemplaza fun1 por tu función real
    
    #suma_extremos = y[0] + y[n]
    #suma_impares = np.sum(y[1:n:2])     # Índices 1, 3, 5... hasta n-1
    #suma_pares = np.sum(y[2:n-1:2])     # Índices 2, 4, 6... hasta n-2

    S = (y[0] + y[n]) + 4 * (np.sum(y[1:n:2])) + 2 * (np.sum(y[2:n-1:2]))
    #S = suma_extremos + 4*suma_impares + 2*suma_pares

    I_simpson = (h / 3) * S
    return I_simpson


print (simpson13(0, np.pi, 100))
#print (trapezoidal(0, np.pi, 100))

