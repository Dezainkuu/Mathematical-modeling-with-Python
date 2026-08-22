import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate


def FPlot(X,Y,Xlbl,Ylbl,Ttl,c,lbl):
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

def F1 (t):
    return 2*np.exp(-t)*np.sin(t+0.5)

def F2 (t):
    return 3.5*np.exp(-7.6*t)*np.cos(3*t-0.3)

def F3 (t):
    return 2.5*np.exp(-8.5*t)*np.sin(1.5*t+np.pi)

t = np.linspace(0, 5, 500)


plt.figure(1) #figure crea ventanas distintas sin bloqueo (no como FPlot)
FPlot(t, F1(t), 't [s]', 'F1(t) [N]', 'F1(t)', 'b', 'F1(t)')

plt.figure(2)
FPlot(t, F2(t), 't [s]', 'F2(t) [N]', 'F2(t)', 'r', 'F2(t)')

plt.figure(3)
FPlot(t, F3(t), 't [s]', 'F3(t) [N]', 'F3(t)', 'g', 'F3(t)')

plt.figure(4)
FPlot(t, F1(t), 't [s]', 'F(t) [N]', 'Las tres funciones', 'b', 'F1(t)')
FPlot(t, F2(t), 't [s]', 'F(t) [N]', 'Las tres funciones', 'r', 'F2(t)')
FPlot(t, F3(t), 't [s]', 'F(t) [N]', 'Las tres funciones', 'g', 'F3(t)')

plt.show()