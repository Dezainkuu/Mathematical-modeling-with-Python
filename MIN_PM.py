from scipy.integrate import quad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotf(X,Y,Xlbl,Ylbl,Ttl,c,lbl):
    #.Donde:
    #.     X   : datos de abcisas
    #.     Y   : datos de ordenadas
    #.     Xlbl: etiqueta del eje X
    #.     Ylbl: etiqueta del eje Y
    #.     Ttl : titulo del grafico
    #.     c   : color de la traza
    #.     lbl : etiqueta de los datos
    plt.xlabel(Xlbl); plt.ylabel(Ylbl); plt.title(Ttl)
    plt.plot(X,Y,c,label=lbl)
    # Grilla mayor
    plt.grid(which='major', color='#666666', linestyle='-')
    # Grilla menor
    plt.minorticks_on()
    plt.grid(which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.legend()
    return

#FUNCIÓN A CALCULAR ----------------------------------------------------    
def fun1(x):
    #f = np.cos(4*x) * np.cos (3 * np.sin(x))
    #f = np.sqrt(4-x**2)/(5-x**2)
    f = x*np.sin(x)
    return f
#------------------------------------------------------------------------

#método de integración por punto medio
def punto_medio (a, b, n):
    h = (b - a)/n  # Ancho de los subintervalos
    x = np.linspace (a+0.5*h, b-0.5*h, n) #n son los puntos, por lo que necesitamos n+1 para los intervalos y el recorrido correcto
    y = fun1 (x)  # Reemplaza fun1 por la función real
    I_pm = h*np.sum(y)
    return I_pm

#método de integración trapezoidal
def trapezoidal (a, b, n):
    h = (b - a)/n
    x = np.linspace (a, b, n+1) 
    y = fun1 (x)
    S = y[0] + 2*np.sum(y[1:n]) + y[n]
    I_trpz = 0.5*h*S
    plotf(x, y, 'x', 'F1(x)', 'F1', 'g', 'F1')
    plt.show()
    return I_trpz


#método de integración simpson 1/3
def simpson13(a, b, n): # El método requiere estrictamente un número par de intervalos
    
    if n % 2 != 0:
        raise ValueError("El número de subintervalos 'n' debe ser un número par.")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = fun1(x)
    
    #suma_extremos = y[0] + y[n]
    #suma_impares = np.sum(y[1:n:2])     # Índices 1, 3, 5... hasta n-1
    #suma_pares = np.sum(y[2:n-1:2])     # Índices 2, 4, 6... hasta n-2

    S = (y[0] + y[n]) + 4 * (np.sum(y[1:n:2])) + 2 * (np.sum(y[2:n-1:2]))
    #S = suma_extremos + 4*suma_impares + 2*suma_pares

    I_simpson = (h / 3) * S
    return I_simpson

#print ("método de integración por trapecio;:", trapezoidal(0, np.pi, 100))
#print ("método de integración por punto medio;:", punto_medio(0, np.pi, 100))
#print ("método de integración por Simpson 1/3;:", simpson13(0, np.pi, 100))

if __name__ == "__main__":
    
    # Parámetros de la integral----------------------------------------------
    a, b, n = 0, np.pi, 100
    valor_exacto = 3.14159265
    #------------------------------------------------------------------------

    resultados = {
        "Punto medio": punto_medio(a, b, n),
        "Trapezoidal": trapezoidal(a, b, n),
        "Simpson 1/3": simpson13(a, b, n),
        "Valor exacto": valor_exacto
    }

    tabla = pd.DataFrame(resultados.items(), columns=["Método", "Valor"])

    # Calcular error porcentual respecto al valor exacto
    tabla["Error (%)"] = (abs(tabla["Valor"] - valor_exacto) / abs(valor_exacto) * 100)
    
    tabla.loc[tabla["Método"] == "Valor exacto", "Error (%)"] = 0 # Si el valor exacto no tiene error
    print("\n", f"Intervalo [a,b] = [{a}, {b}]   n = {n}\n")
    print("-" * 55, "\n", tabla.to_string(index=False, float_format="%.6f", col_space=18),"\n", "-" * 55)
