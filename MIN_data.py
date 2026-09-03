import pandas as pd

# Datos del problema
x = [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00]
y = [2.7183, 3.0042, 3.3201, 3.6693, 4.0552, 4.4817, 4.9530, 5.4739, 6.0496, 6.6859, 7.3891]
exact = 4.670774

# Cálculos de integración
h = 0.1
I_trap = (h / 2) * (y[0] + 2 * sum(y[1:-1]) + y[-1])
I_simp = (h / 3) * (y[0] + 4 * sum(y[1:-1:2]) + 2 * sum(y[2:-1:2]) + y[-1])

h_pm = 0.2
I_pm = h_pm * sum(y[1::2])

# Estructura de datos
resultados = {
    "Método": ["Punto medio", "Trapezoidal", "Simpson 1/3", "Valor exacto"],
    "Valor": [I_pm, I_trap, I_simp, exact]
}

tabla = pd.DataFrame(resultados) #Creación del DataFrame de Pandas
tabla["Error (%)"] = abs(tabla["Valor"] - exact) / exact * 100 #Cálculo del error porcentual (%)
print("-" * 55, "\n", tabla.to_string(index=False, float_format="%.6f", col_space=18), "\n", "-" * 55, "\n") #Presentación del DataFrame