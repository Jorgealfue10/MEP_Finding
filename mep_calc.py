import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
from typing import Tuple, Union

# Añadir rutas a tus módulos personalizados
sys.path.append('/home/jorgebdelafuente/Codes/MEP_Finding/')
from PHMGS import *
from PHM1QSm import *
from H2_diat import *
import fitGS
import fit1QSm
import fitpy8
import SAp10
import SApp10

Number = Union[float, int, np.ndarray]

# -----------------------------------------------------------------------------
# Funciones de energía potencial por tipo de PES
# -----------------------------------------------------------------------------
def PES(coords, tipo="GS"):
    r_PH1, r_PH2, r_HH = coords
    if tipo == "GS":
        return energia1TApp(r_PH1, r_PH2, r_HH)
    elif tipo == "1QSm":
        return energia2TApp(r_PH1, r_PH2, r_HH)
    elif tipo == "1TAp":
        return energia1TAp(r_PH1, r_PH2, r_HH)
    elif tipo == "1SAp":
        return energia1SAp(r_PH1, r_PH2, r_HH)
    elif tipo == "1SApp":
        return energia1SApp(r_PH1, r_PH2, r_HH)
    else:
        raise ValueError(f"PES tipo desconocido: {tipo}")

def energia1TApp(r_PH1, r_PH2, r_HH):
    e12, _ = diatphm(r_PH1)
    e13, _ = diatphm(r_PH2)
    e23, _ = diathh(r_HH)
    bod3, _ = fitGS.fit3d(r_PH1, r_PH2, r_HH)
    return bod3 + e12 + e13 + e23

def energia2TApp(r_PH1, r_PH2, r_HH):
    e12, _ = diat1qsm(r_PH1)
    e13, _ = diat1qsm(r_PH2)
    e23, _ = diathh(r_HH)
    bod3, _ = fit1QSm.fit3d(r_PH1, r_PH2, r_HH)
    return bod3 + e12 + e13 + e23

def energia1TAp(r_PH1, r_PH2, r_HH):
    e12, _ = diatphm(r_PH1)
    e13, _ = diatphm(r_PH2)
    e23, _ = diathh(r_HH)
    dummy_grad = [0, 0, 0]
    bod3 = fitpy8.fit3d(r_PH1, r_PH2, r_HH, dummy_grad)
    return bod3 + e12 + e13 + e23

# h2p1d.py

def load_data(
    coeffs_path: str | Path,
    nodes_path:  str | Path,
) -> None:
    """Carga los ficheros de coeficientes (ncoeffs x 4) y nodos (nnodes)."""
    global _coeffs, _nodes
    coeffs_path = Path(coeffs_path)
    nodes_path  = Path(nodes_path)

    _coeffs = np.loadtxt(coeffs_path, dtype=np.float64)   # (ncoeffs, 4)
    _nodes  = np.loadtxt(nodes_path,  dtype=np.float64)   # (nnodes,)
    if _coeffs.ndim != 2 or _coeffs.shape[1] != 4:
        raise ValueError("coeffs debe ser (ncoeffs, 4)")
    if _nodes.ndim != 1 or _nodes.size != _coeffs.shape[0] + 1:
        raise ValueError("nnodes debe ser ncoeffs + 1")
    # Asegura orden C para operaciones rápidas
    _coeffs = np.ascontiguousarray(_coeffs, dtype=np.float64)
    _nodes  = np.ascontiguousarray(_nodes,  dtype=np.float64)

def _ensure_loaded():
    if _coeffs is None or _nodes is None:
        raise RuntimeError("Debes llamar primero a load_data(coeffs_path, nodes_path)")

def _find_intervals(x: np.ndarray) -> np.ndarray:
    """
    Devuelve el índice i de intervalo tal que nodes[i] <= x < nodes[i+1].
    Para x fuera de rango, devuelve -1.
    """
    # searchsorted da la posición de inserción a la derecha de nodes[i], usamos -1
    idx = np.searchsorted(_nodes, x, side='right') - 1
    # fuera de rango -> -1
    idx[(x < _nodes[0]) | (x > _nodes[-1])] = -1
    return idx

def spl_eval(x: Number) -> np.ndarray:
    """Evalúa el cúbico trozo a trozo; fuera de rango devuelve 0.0 (como tu Fortran)."""
    _ensure_loaded()
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.zeros_like(x_arr, dtype=np.float64)

    idx = _find_intervals(x_arr)
    valid = idx >= 0
    if np.any(valid):
        i = idx[valid]
        a = _coeffs[i, 0]
        b = _coeffs[i, 1]
        c = _coeffs[i, 2]
        d = _coeffs[i, 3]
        dx = x_arr[valid] - _nodes[i]
        y[valid] = a + b*dx + c*dx*dx + d*dx*dx*dx
    return y if isinstance(x, np.ndarray) else y.item()

def spl_deriv(x: Number) -> np.ndarray:
    """Derivada del cúbico; fuera de rango devuelve 0.0 (como tu Fortran)."""
    _ensure_loaded()
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    dy = np.zeros_like(x_arr, dtype=np.float64)

    idx = _find_intervals(x_arr)
    valid = idx >= 0
    if np.any(valid):
        i = idx[valid]
        b = _coeffs[i, 1]
        c = _coeffs[i, 2]
        d = _coeffs[i, 3]
        dx = x_arr[valid] - _nodes[i]
        dy[valid] = b + 2.0*c*dx + 3.0*d*dx*dx
    return dy if isinstance(x, np.ndarray) else dy.item()

def diatHH_sing_py(r: Number) -> Tuple[np.ndarray, np.ndarray]:
    """
    Réplica de tu subrutina Fortran:
      subroutine diatHH_sing(r, ener, der)
    Devuelve (ener, der). Acepta escalar o array; siempre float64.
    """
    ener = spl_eval(r)
    der  = spl_deriv(r)
    return ener, der

load_data(
    "/home/jorgebdelafuente/Doctorado/RKHS/H2Psing-OK/cubic_spl_notrkhs/spline_coeffs.txt",
    "/home/jorgebdelafuente/Doctorado/RKHS/H2Psing-OK/cubic_spl_notrkhs/spline_nodes.txt",
)

def energia1SAp(r_PH1, r_PH2, r_HH):
    e12, _ = diatphm(r_PH1)
    e13, _ = diatphm(r_PH2)
    e23, _ = diatHH_sing_py(r_HH)
    bod3, _ = SAp10.fit3d(r_PH1, r_PH2, r_HH)
    return bod3 + e12 + e13 + e23

def energia1SApp(r_PH1, r_PH2, r_HH):
    e12, _ = diatphm(r_PH1)
    e13, _ = diatphm(r_PH2)
    e23, _ = diatHH_sing_py(r_HH)
    bod3, _ = SApp10.fit3d(r_PH1, r_PH2, r_HH)
    return bod3 + e12 + e13 + e23

# -----------------------------------------------------------------------------
# Gradiente numérico por diferencias finitas
# -----------------------------------------------------------------------------
def numerical_gradient(coords, tipo, h=1e-5):
    grad = np.zeros_like(coords)
    for i in range(len(coords)):
        x1 = coords.copy()
        x2 = coords.copy()
        x1[i] += h
        x2[i] -= h
        f1 = PES(x1, tipo)
        f2 = PES(x2, tipo)
        grad[i] = (f1 - f2) / (2 * h)
    return grad

# -----------------------------------------------------------------------------
# Steepest Descent con momentum
# -----------------------------------------------------------------------------
def steepest_descent_momentum(start_coords, max_steps, tipo, step_size=0.05, tol=1e-10, momentum=0.85):
    path = [start_coords]
    coords = np.array(start_coords)
    prev_coords = np.copy(coords)

    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
        grad = numerical_gradient(coords, tipo)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        update = -step_size * grad + momentum * (coords - prev_coords)
        prev_coords = np.copy(coords)
        coords = coords + update
        path.append(coords)

    return np.array(path)

# -----------------------------------------------------------------------------
# Interpolación suave con corrección por gradiente
# -----------------------------------------------------------------------------
def smooth_interpolation_on_pes(start_coords, end_coords, n_steps, tipo, energy_threshold=0.2):
    start = np.array(start_coords)
    end = np.array(end_coords)
    path, energies = [], []

    for i in tqdm(range(n_steps + 1), desc="Interpolating", ncols=100, unit="step"):
        alpha = i / n_steps
        guess = (1 - alpha) * start + alpha * end
        E = PES(guess, tipo)

        if i > 0 and abs(E - energies[-1]) > energy_threshold:
            grad = numerical_gradient(guess, tipo)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                guess = guess - 0.05 * grad / grad_norm
                E = PES(guess, tipo)

        path.append(guess)
        energies.append(E)

    return np.array(path), np.array(energies)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MEP optimization script")
    parser.add_argument("--type", choices=["GS", "1QSm", "1TAp", "1SAp", "1SApp"], default="GS",
                        help="Tipo de superficie PES: 'GS', '1QSm' o '1TAp'")
    parser.add_argument("--output", default="minimum_energy_path.dat",
                        help="Nombre del archivo de salida")
    args = parser.parse_args()
    tipo = args.type

    print(f"Usando superficie PES tipo: {tipo}")

    start1 = np.array([12.0, 12.0, 1.40])
    start2 = np.array([2.7, 12.0, 12.0])
    

    # Calcular caminos
    if tipo == "GS":
        path1 = steepest_descent_momentum(start1, 1000000, tipo)
        path2 = steepest_descent_momentum(start2, 100000, tipo)[::-1]
        path_middle, _ = smooth_interpolation_on_pes(path1[-1], path2[0], 100000, tipo)

        E1 = np.array([PES(coords, tipo) for coords in path1])
        Em = np.array([PES(coords, tipo) for coords in path_middle])
        E2 = np.array([PES(coords, tipo) for coords in path2])
        i_min1, i_max1 = np.argmin(E1), np.argmax(E1)
        E1_min, E1_max = E1[i_min1], E1[i_max1]
        i_min2, i_max2 = np.argmin(E2), np.argmax(E2)
        E2_min, E2_max = E2[i_min2], E2[i_max2]
        i_minm, i_maxm = np.argmin(Em), np.argmax(Em)
        Em_min, Em_max = Em[i_minm], Em[i_maxm]
        geom_min1, geom_max1 = path1[i_min1], path1[i_max1]
        geom_min2, geom_max2 = path2[i_min2], path2[i_max2]
        geom_minm, geom_maxm = path_middle[i_minm], path_middle[i_maxm]

        final_path = np.vstack([path1, path_middle, path2])
    elif tipo == "1QSm":
        geom_equil = np.array([6.040695, 6.040695, 1.400000])  
        geom_equil2 = np.array([2.80000000, 2.83139235, 3.16224475])  
        path1, _ = smooth_interpolation_on_pes(start1, geom_equil, 1000, tipo)
        path2, _ = smooth_interpolation_on_pes(start2, geom_equil2, 1000, tipo)
        bridge_path, _ = smooth_interpolation_on_pes(geom_equil,geom_equil2, 10000, tipo)

        E1 = np.array([PES(coords, tipo) for coords in path1])
        Em = np.array([PES(coords, tipo) for coords in bridge_path])
        E2 = np.array([PES(coords, tipo) for coords in path2])
        i_min1, i_max1 = np.argmin(E1), np.argmax(E1)
        E1_min, E1_max = E1[i_min1], E1[i_max1]
        i_min2, i_max2 = np.argmin(E2), np.argmax(E2)
        E2_min, E2_max = E2[i_min2], E2[i_max2]
        i_minm, i_maxm = np.argmin(Em), np.argmax(Em)
        Em_min, Em_max = Em[i_minm], Em[i_maxm]
        geom_min1, geom_max1 = path1[i_min1], path1[i_max1]
        geom_min2, geom_max2 = path2[i_min2], path2[i_max2]
        geom_minm, geom_maxm = bridge_path[i_minm], bridge_path[i_maxm]

        final_path = np.vstack([path1, bridge_path, path2[::-1]])  
    elif tipo == "1TAp":
        geom_equil = np.array([6.040695, 6.040695, 1.400000])  
        geom_equil2 = np.array([2.80000000, 2.83139235, 3.16224475])  
        path1, _ = smooth_interpolation_on_pes(start1, geom_equil, 1000, tipo)
        path2, _ = smooth_interpolation_on_pes(start2, geom_equil2, 1000, tipo)
        bridge_path, _ = smooth_interpolation_on_pes(geom_equil, geom_equil2, 10000, tipo)

        E1 = np.array([PES(coords, tipo) for coords in path1])
        Em = np.array([PES(coords, tipo) for coords in bridge_path])
        E2 = np.array([PES(coords, tipo) for coords in path2])
        i_min1, i_max1 = np.argmin(E1), np.argmax(E1)
        E1_min, E1_max = E1[i_min1], E1[i_max1]
        i_min2, i_max2 = np.argmin(E2), np.argmax(E2)
        E2_min, E2_max = E2[i_min2], E2[i_max2]
        i_minm, i_maxm = np.argmin(Em), np.argmax(Em)
        Em_min, Em_max = Em[i_minm], Em[i_maxm]
        geom_min1, geom_max1 = path1[i_min1], path1[i_max1]
        geom_min2, geom_max2 = path2[i_min2], path2[i_max2]
        geom_minm, geom_maxm = bridge_path[i_minm], bridge_path[i_maxm]

        final_path = np.vstack([path1, bridge_path, path2[::-1]])
    elif tipo == "1SAp":
        path1 = steepest_descent_momentum(start1, 1000000, tipo)
        path2 = steepest_descent_momentum(start2, 100000, tipo)[::-1]
        path_middle, _ = smooth_interpolation_on_pes(path1[-1], path2[0], 100000, tipo)
        final_path = np.vstack([path1, path_middle, path2])

        E1 = np.array([PES(coords, tipo) for coords in path1])
        Em = np.array([PES(coords, tipo) for coords in path_middle])
        E2 = np.array([PES(coords, tipo) for coords in path2])
        i_min1, i_max1 = np.argmin(E1), np.argmax(E1)
        E1_min, E1_max = E1[i_min1], E1[i_max1]
        i_min2, i_max2 = np.argmin(E2), np.argmax(E2)
        E2_min, E2_max = E2[i_min2], E2[i_max2]
        i_minm, i_maxm = np.argmin(Em), np.argmax(Em)
        Em_min, Em_max = Em[i_minm], Em[i_maxm]
        geom_min1, geom_max1 = path1[i_min1], path1[i_max1]
        geom_min2, geom_max2 = path2[i_min2], path2[i_max2]
        geom_minm, geom_maxm = path_middle[i_minm], path_middle[i_maxm]

    elif tipo == "1SApp":
        geom_equil = np.array([3.110048, 3.110048, 1.640000])  
        geom_equil2 = np.array([2.70000000, 2.64611218, 4.81708436])  
        path1, _ = smooth_interpolation_on_pes(start1, geom_equil, 1000, tipo)
        path2, _ = smooth_interpolation_on_pes(start2, geom_equil2, 1000, tipo)
        bridge_path, _ = smooth_interpolation_on_pes(geom_equil, geom_equil2, 10000, tipo)

        E1 = np.array([PES(coords, tipo) for coords in path1])
        Em = np.array([PES(coords, tipo) for coords in bridge_path])
        E2 = np.array([PES(coords, tipo) for coords in path2])
        i_min1, i_max1 = np.argmin(E1), np.argmax(E1)
        E1_min, E1_max = E1[i_min1], E1[i_max1]
        i_min2, i_max2 = np.argmin(E2), np.argmax(E2)
        E2_min, E2_max = E2[i_min2], E2[i_max2]
        i_minm, i_maxm = np.argmin(Em), np.argmax(Em)
        Em_min, Em_max = Em[i_minm], Em[i_maxm]
        geom_min1, geom_max1 = path1[i_min1], path1[i_max1]
        geom_min2, geom_max2 = path2[i_min2], path2[i_max2]
        geom_minm, geom_maxm = bridge_path[i_minm], bridge_path[i_maxm]

        final_path = np.vstack([path1, bridge_path, path2])


    # Evaluar energía
    path_with_energy = []
    for coords in final_path:
        energy = PES(coords, tipo)
        path_with_energy.append(np.append(coords, energy))
    path_with_energy = np.array(path_with_energy)

    # Guardar resultados
    np.savetxt(args.output, path_with_energy[::2], header="r1 r2 r3 energy")
    print(f"Ruta guardada en {args.output}")

    print(f"Geom. minima 1: {geom_min1}\nGeom. maxima 1: {geom_max1}\nEnergia minima 1: {E1_min}\nEnergia maxima 1: {E1_max}")
    print(f"Geom. minima 2: {geom_min2}\nGeom. maxima 2: {geom_max2}\nEnergia minima 2: {E2_min}\nEnergia maxima 2: {E2_max}")
    print(f"Geom. minima m: {geom_minm}\nGeom. maxima m: {geom_maxm}\nEnergia minima m: {Em_min}\nEnergia maxima m: {Em_max}")

    # Gráfica
    _, _, _, energy = path_with_energy.T
    plt.figure(figsize=(8, 5))
    plt.plot(energy, color="black")
    plt.xlabel("Paso en el MEP")
    plt.ylabel("Energía (u.a.)")
    plt.title(f"Evolución energética – tipo: {tipo}")
    delta_E = (energy[-1] - energy[0]) * 27.2114
    plt.annotate(f"$\\Delta E$ = {delta_E:.2f} eV", xy=(0.5, 0.95), xycoords="axes fraction",
                ha="center", va="top", size=12)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
