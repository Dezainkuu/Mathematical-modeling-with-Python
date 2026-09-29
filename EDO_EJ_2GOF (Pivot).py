"""
Sistema de 2 GDL: barra rigida (m, J) pivotada en su centro O, con resortes-
amortiguadores directos a tierra en los puntos A (izquierda, k1,c1) y B
(derecha, k2,c2). Sin masas colgantes (version simple).
 
GDL: y (traslacion del pivote/CM), theta (rotacion). Convencion: +y hacia
arriba, +theta horario (ver DCL). Estado: X = [y, y', theta, theta']
 
Puntos de la barra: yA = y + L1*theta ; yB = y - L2*theta
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# ===========================================================================
# 1) PARAMETROS 
# ===========================================================================
params = dict(
    m=5.0, J=1.5,
    k1=200.0, c1=8.0,
    k2=150.0, c2=6.0,
    L1=0.4, L2=0.6,
)