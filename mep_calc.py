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

# -----------------------------------------------------------------------------
# Soporte spline H2 singlete (tu réplica Fortran)
# -----------------------------------------------------------------------------
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
    idx = np.searchsorted(_nodes, x, side='right') - 1
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
# Gradiente numérico standard (3D: r12, r13, r23)
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
# Steepest Descent con momentum (3D)
# -----------------------------------------------------------------------------
def steepest_descent_momentum(start_coords, max_steps, tipo, step_size=0.05, tol=1e-10, momentum=0.85):
    path = [start_coords]
    coords = np.array(start_coords, dtype=float)
    prev_coords = coords.copy()

    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
        grad = numerical_gradient(coords, tipo)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol or not np.isfinite(grad_norm):
            break

        update = -step_size * grad + momentum * (coords - prev_coords)
        prev_coords = coords.copy()
        coords = coords + update
        path.append(coords.copy())

    return np.array(path)

# -----------------------------------------------------------------------------
# Interpolación suave con corrección por gradiente (3D)
# -----------------------------------------------------------------------------
def smooth_interpolation_on_pes(start_coords, end_coords, n_steps, tipo, energy_threshold=0.2):
    start = np.array(start_coords, dtype=float)
    end = np.array(end_coords, dtype=float)
    path, energies = [], []

    for i in tqdm(range(n_steps + 1), desc="Interpolating", ncols=100, unit="step"):
        alpha = i / n_steps
        guess = (1 - alpha) * start + alpha * end
        E = PES(guess, tipo)

        if i > 0 and abs(E - energies[-1]) > energy_threshold:
            grad = numerical_gradient(guess, tipo)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0 and np.isfinite(grad_norm):
                guess = guess - 0.05 * grad / grad_norm
                E = PES(guess, tipo)

        path.append(guess)
        energies.append(E)

    return np.array(path), np.array(energies)

# ======================= MODO θ: helpers (INTERNAS r12,r23) ===================
def r13_from_internals(r12: float, r23: float, theta_deg: float) -> float | None:
    th = np.radians(theta_deg)
    arg = r12*r12 + r23*r23 + 2.0*r12*r23*np.cos(th)
    if arg <= 0.0:
        return None
    return np.sqrt(arg)

def PES_theta(coords2d, tipo: str, theta_deg: float) -> float:
    """coords2d = (r12, r23); θ fijo. Devuelve energía (Hartree)."""
    r12, r23 = float(coords2d[0]), float(coords2d[1])
    r13 = r13_from_internals(r12, r23, theta_deg)
    if r13 is None:
        return np.inf
    return PES((r12, r13, r23), tipo)

def numerical_gradient_theta(coords2d, tipo: str, theta_deg: float, h=1e-5):
    r12, r23 = float(coords2d[0]), float(coords2d[1])
    f1 = PES_theta((r12+h, r23), tipo, theta_deg)
    f2 = PES_theta((r12-h, r23), tipo, theta_deg)
    g12 = (f1 - f2) / (2*h)
    f1 = PES_theta((r12, r23+h), tipo, theta_deg)
    f2 = PES_theta((r12, r23-h), tipo, theta_deg)
    g23 = (f1 - f2) / (2*h)
    return np.array([g12, g23])

def steepest_descent_momentum_theta(start_coords2d, max_steps, tipo, theta_deg,
                                    step_size=0.05, tol=1e-10, momentum=0.85):
    """Steepest descent con momentum en (r12,r23) con θ fijo (r13 via ley de cosenos)."""
    path = [np.array(start_coords2d, dtype=float)]
    coords = np.array(start_coords2d, dtype=float)
    prev_coords = coords.copy()

    for _ in tqdm(range(max_steps), desc=f"Optimizing θ={theta_deg:g}", ncols=100, unit="step"):
        grad = numerical_gradient_theta(coords, tipo, theta_deg)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol or not np.isfinite(grad_norm):
            break

        update = -step_size * grad + momentum * (coords - prev_coords)
        prev_coords = coords.copy()
        coords = coords + update
        path.append(coords.copy())

    return np.array(path)

def smooth_interpolation_on_pes_theta(start_coords2d, end_coords2d, n_steps, tipo, theta_deg,
                                      energy_threshold=0.2):
    """Interpolación “theta-aware” en (r12,r23)."""
    start = np.array(start_coords2d, dtype=float)
    end   = np.array(end_coords2d,   dtype=float)
    path, energies = [], []
    for i in tqdm(range(n_steps + 1), desc=f"Interpol θ={theta_deg:g}", ncols=100, unit="step"):
        alpha = i / n_steps
        guess = (1 - alpha) * start + alpha * end
        E = PES_theta(guess, tipo, theta_deg)
        if i > 0 and abs(E - energies[-1]) > energy_threshold:
            g = numerical_gradient_theta(guess, tipo, theta_deg)
            gn = np.linalg.norm(g)
            if gn > 0 and np.isfinite(gn):
                guess = guess - 0.05 * g / gn
                E = PES_theta(guess, tipo, theta_deg)
        path.append(guess)
        energies.append(E)
    return np.array(path), np.array(energies)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MEP optimization script")
    parser.add_argument("--type", choices=["GS", "1QSm", "1TAp", "1SAp", "1SApp"], default="GS",
                        help="Tipo de superficie PES: 'GS', '1QSm', '1TAp', '1SAp', '1SApp'")
    parser.add_argument("--output", default="minimum_energy_path.dat",
                        help="Nombre del archivo de salida")
    # NUEVO: si se especifica, activa el modo θ (internas r12,r23, r13 por ley de cosenos)
    parser.add_argument("--theta", type=float, default=None,
                        help="Ángulo P–H–H fijo (grados). Si se omite, se usa el flujo original 3D.")
    args = parser.parse_args()
    tipo = args.type

    # ======================= MODO θ FIJO (alternativo) =======================
    if args.theta is not None:
        theta = float(args.theta)
        print(f"[θ-modo] Tipo: {tipo} | θ = {theta:.1f}° (internas r12,r23; r13 por ley de cosenos)")

        # Mismos starts que ya tenías, proyectados a internas (r12, r23)
        start1_3d = np.array([12.0, 12.0, 1.40], dtype=float)
        start2_3d = np.array([2.7, 12.0, 12.0], dtype=float)
        start_in_2d  = np.array([start1_3d[0], start1_3d[2]], dtype=float)  # (r12, r23)
        start_out_2d = np.array([start2_3d[0], start2_3d[2]], dtype=float)  # (r12, r23)

        # Para tipos que en tu original usan SD + interpolación:
        if tipo in ["GS", "1SAp"]:
            path1_2d = steepest_descent_momentum_theta(start_in_2d,  max_steps=1_000_000,
                                                       tipo=tipo, theta_deg=theta)
            path2_2d = steepest_descent_momentum_theta(start_out_2d, max_steps=100_000,
                                                       tipo=tipo, theta_deg=theta)[::-1]
            path_mid_2d, _ = smooth_interpolation_on_pes_theta(path1_2d[-1], path2_2d[0],
                                                               100_000, tipo, theta_deg=theta)
            final_path_2d = np.vstack([path1_2d, path_mid_2d, path2_2d])

        # Para tipos que en tu original usan sólo interpolaciones entre geometrías
        elif tipo in ["1QSm", "1TAp"]:
            geom_equil  = np.array([6.040695, 6.040695, 1.400000], dtype=float)
            geom_equil2 = np.array([2.80000000, 2.83139235, 3.16224475], dtype=float)
            # proyectamos a (r12, r23) – r13 se ajustará con θ en la evaluación
            g1_2d = np.array([geom_equil[0],  geom_equil[2]],  dtype=float)
            g2_2d = np.array([geom_equil2[0], geom_equil2[2]], dtype=float)
            path1_2d, _ = smooth_interpolation_on_pes_theta(start_in_2d, g1_2d, 1000, tipo, theta)
            path2_2d, _ = smooth_interpolation_on_pes_theta(start_out_2d, g2_2d, 1000, tipo, theta)
            bridge_2d, _ = smooth_interpolation_on_pes_theta(g1_2d, g2_2d, 10000, tipo, theta)
            final_path_2d = np.vstack([path1_2d, bridge_2d, path2_2d[::-1]])

        elif tipo == "1SApp":
            geom_equil  = np.array([3.110048, 3.110048, 1.640000], dtype=float)
            geom_equil2 = np.array([2.70000000, 2.64611218, 4.81708436], dtype=float)
            g1_2d = np.array([geom_equil[0],  geom_equil[2]],  dtype=float)
            g2_2d = np.array([geom_equil2[0], geom_equil2[2]], dtype=float)
            path1_2d, _ = smooth_interpolation_on_pes_theta(start_in_2d, g1_2d, 1000, tipo, theta)
            path2_2d, _ = smooth_interpolation_on_pes_theta(start_out_2d, g2_2d, 1000, tipo, theta)
            bridge_2d, _ = smooth_interpolation_on_pes_theta(g1_2d, g2_2d, 10000, tipo, theta)
            final_path_2d = np.vstack([path1_2d, bridge_2d, path2_2d])

        else:
            raise ValueError(f"Tipo de PES no soportado en θ-modo: {tipo}")

        # Reconstrucción (r12, r13(θ), r23) + energía
        path_with_energy = []
        for r12, r23 in final_path_2d:
            r13 = r13_from_internals(r12, r23, theta)
            if r13 is None:
                continue
            E = PES((r12, r13, r23), tipo)
            path_with_energy.append([r12, r13, r23, E])
        path_with_energy = np.array(path_with_energy, dtype=float)

        # Métricas globales (como al final de tu script)
        E_all = path_with_energy[:, -1]
        i_min, i_max = np.argmin(E_all), np.argmax(E_all)
        geom_min  = path_with_energy[i_min, :3]
        geom_max  = path_with_energy[i_max, :3]
        Emin, Emax = E_all[i_min], E_all[i_max]

        # Guardar (submuestreo como hacías)
        np.savetxt(args.output, path_with_energy[::2], header="r12 r13 r23 energy")
        print(f"Ruta (θ={theta:.0f}°) guardada en {args.output}")
        print(f"Geom. mínima: {geom_min}   Energía mínima: {Emin}")
        print(f"Geom. máxima: {geom_max}   Energía máxima: {Emax}")

        # Plot energía en u.a. y ΔE en eV
        _, _, _, energy = path_with_energy.T
        plt.figure(figsize=(8, 5))
        plt.plot(energy, color="black")
        plt.xlabel("Paso en el MEP (θ fijo)")
        plt.ylabel("Energía (u.a.)")
        plt.title(f"Evolución energética – tipo: {tipo} – θ={theta:.0f}°")
        delta_E = (energy[-1] - energy[0]) * 27.2114
        plt.annotate(f"$\\Delta E$ = {delta_E:.2f} eV", xy=(0.5, 0.95), xycoords="axes fraction",
                    ha="center", va="top", size=12)
        plt.grid()
        plt.show()
        return
    # ===================== FIN MODO θ — MODO ORIGINAL ABAJO =====================

    print(f"Usando superficie PES tipo: {tipo}")

    start1 = np.array([12.0, 12.0, 1.40])
    start2 = np.array([2.7, 12.0, 12.0])

    # Calcular caminos (flujo original intacto)
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
