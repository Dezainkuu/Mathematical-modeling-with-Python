import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

matriz_a = np.array([[3, 7, 1, 4], [3, 1, 2, 8], [9, 8, 4, 5], [5, 6, 7, 1]])

matriz_b = matriz_a.T #traspuesta de la matriz_a

matriz_c = matriz_a*matriz_b #multiplicacion de la matriz_a por su traspuesta elemento a elemento

m_suma = matriz_a + matriz_b #suma de la matriz_a y su traspuesta

m_div = matriz_a/matriz_b #division de la matriz_a entre su traspuesta elemento a elemento

Mpab = matriz_a @ matriz_b #multiplicacion matricial de la matriz_a por su traspuesta

Mdab = matriz_a@np.linalg.inv(matriz_b) #DIVISION MATRICIAL [multiplicacion matricial de la matriz_a por la inversa de su traspuesta]


#print (matriz_a [2, 2], "\n") #imprime el elemento de la fila 2 y columna 3 de la matriz_a

#print (matriz_a, "\n")
#print (matriz_b, "\n")
#print (matriz_c)
#print ((matriz_a @ matriz_b), "\n")

matriz_d = matriz_a.copy() #copia de la matriz_a a la matriz_d
matriz_d [1:3,1:3] = 1 #cambio de los elementos centrales de la matriz_a por 1

#print (matriz_a, "\n")

#--------------------------------------------------------------------------------------------------------------------------------

Vf = matriz_a.diagonal() #obtiene los elementos de la diagonal de la matriz_a
print (Vf, "\n")
#print (matriz_a, "\n")

#--------------------------------------------------------------------------------------------------------------------------------

Vc = matriz_a[:, 1]
Vc = Vc.reshape((4, 1)) #obtiene la segunda columna de la matriz_a y la convierte en un vector columna de 4 filas y 1 columna
print (Vc, "\n")

#Vc = np.zeros((4, 1)) #vector columna de ceros de 4 filas y 1 columna
#Vc = matriz_a[:, [1]] #obtiene la segunda columna de la matriz_a

#--------------------------------------------------------------------------------------------------------------------------------
print (Vc.T@Vf) #Multiplicación escalar entre vectores

#--------------------------------------------------------------------------------------------------------------------------------

F1 = matriz_a[0, :] #obtiene la primera fila de la matriz_a
C1 = matriz_a[:, 0] #obtiene la primera columna de la matriz_a

F1c = F1.reshape((1, 4)) #convierte la primera fila en un vector fila de 1 fila y 4 columnas
C1c = C1.reshape((4, 1)) #convierte la primera columna en un vector columna de 4 filas y 1 columna

Ang = np.arange(5, 181, 5) #vector de ángulos de 5 a 180 grados con incrementos de 5 grados

print (Ang, "\n")

Ang = np.insert(Ang, 0, 0) #inserta un cero al inicio del vector Ang

print (Ang, "\n")

PAng = Ang [Ang%2 == 0] #obtiene los ángulos pares del vector Ang
print (PAng, "\n")

PAng = Ang [Ang%2 != 0] #obtiene los ángulos impares del vector Ang
print (PAng, "\n")