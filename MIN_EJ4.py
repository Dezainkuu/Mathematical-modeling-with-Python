import sympy as sp
import numpy as np
from scipy.integrate import quad

x = sp.symbols('x')

def obtener_n(f_expr, a, b, eps=0.001, puntos_busqueda=200_000):

    #Calcula el n minimo para trapecio y Simpson tal que el error teorico sea < eps, en [a,b].
    #f_expr: expresion simbolica de sympy en la variable x.

    f2 = sp.diff(f_expr, x, 2)   # f''  -> para trapecio
    f4 = sp.diff(f_expr, x, 4)   # f'''' -> para simpson

    f2_num = sp.lambdify(x, f2, 'numpy')
    f4_num = sp.lambdify(x, f4, 'numpy')

    xs = np.linspace(a, b, puntos_busqueda) # M2 = max|f''| y M4 = max|f''''| en [a,b], por busqueda numerica fina
    M2 = np.max(np.abs(f2_num(xs)))
    M4 = np.max(np.abs(f4_num(xs)))

    n_trap = int(np.ceil(np.sqrt((b - a)**3 * M2 / (12 * eps))))

    n_simp = int(np.ceil(((b - a)**5 * M4 / (180 * eps)) ** 0.25))
    if n_simp % 2 != 0:      # Simpson exige n par
        n_simp += 1

    return {"f2": f2, "f4": f4, "M2": M2, "M4": M4,
            "n_trapecio": n_trap, "n_simpson": n_simp}


def trapezoidal(f, a, b, n):
    h = (b - a) / n
    xi = np.linspace(a, b, n + 1)
    y = f(xi)
    return 0.5 * h * (y[0] + y[n] + 2 * np.sum(y[1:n]))

def simpson13(f, a, b, n):
    h = (b - a) / n
    xi = np.linspace(a, b, n + 1)
    y = f(xi)
    S = (y[0] + y[n]) + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n - 1:2])
    return (h / 3) * S

if __name__ == "__main__":
    # ---- ejercicio: f(x) = 2x^5 - 3x^3 + 5, en [1,3], eps=0.001 ----
    f_sym = 2*x**5 - 3*x**3 + 5
    a, b, eps = 1.0, 3.0, 0.001

    r = obtener_n(f_sym, a, b, eps)
    print("f''  =", r["f2"], "  -> M2 =", r["M2"])
    print("f'''' =", r["f4"], "  -> M4 =", r["M4"])
    print("n trapecio :", r["n_trapecio"])
    print("n simpson  :", r["n_simpson"])

    f_num = sp.lambdify(x, f_sym, 'numpy')
    exacto, _ = quad(f_num, a, b)

    I_t = trapezoidal(f_num, a, b, r["n_trapecio"])
    I_s = simpson13(f_num, a, b, r["n_simpson"])

    print(f"\nValor exacto: {exacto:.10f}")
    print(f"Trapecio  n={r['n_trapecio']}: I={I_t:.10f}  error={abs(I_t-exacto):.2e}")
    print(f"Simpson   n={r['n_simpson']}: I={I_s:.10f}  error={abs(I_s-exacto):.2e}")