#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
from typing import Tuple, Union

# ===================== Importa tus módulos existentes =====================
sys.path.append('/home/jorgebdelafuente/Codes/MEP_Finding/')
from PHMGS import *
from PHM1QSm import *
from H2_diat import *
import fitGS
import fit1QSm
import fitpy10
import SAp10
import SApp10

Number = Union[float, int, np.ndarray]
HARTREE_TO_EV = 27.2114

# ===================== Control global de θ fija =====================
GLOBAL_THETA: float | None = None  # Se setea en main() con args.theta (o None)

def r13_from_theta(r12: float, r23: float, theta_deg: float) -> float | None:
    """Ley de cosenos (convención +): r13^2 = r12^2 + r23^2 + 2 r12 r23 cosθ"""
    th = np.radians(theta_deg)
    arg = r12*r12 + r23*r23 + 2.0*r12*r23*np.cos(th)
    if arg <= 0.0:
        return None
    return float(np.sqrt(arg))

def project_to_theta(coords: np.ndarray) -> np.ndarray:
    """coords=[r12, r13, r23]; si GLOBAL_THETA está activa, reimpone r13 por ley de cosenos."""
    if GLOBAL_THETA is None:
        return np.array(coords, dtype=float)
    r12, _, r23 = map(float, coords)
    r13 = r13_from_theta(r12, r23, GLOBAL_THETA)
    if r13 is None:
        return np.array([r12, np.nan, r23], dtype=float)
    return np.array([r12, r13, r23], dtype=float)

# -----------------------------------------------------------------------------
# PES por tipo (internas r12, r13, r23)
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

def PES_eval(coords, tipo="GS"):
    """
    Evaluación de la PES respetando θ fijo si GLOBAL_THETA está definida.
    - Sin θ: evalúa PES(coords, tipo) tal cual.
    - Con θ: proyecta coords al colector θ=cte (recalcula r13) y evalúa.
    """
    if GLOBAL_THETA is None:
        return PES(coords, tipo)
    xyz = project_to_theta(coords)
    if not np.isfinite(xyz[1]):
        return np.inf
    return PES(xyz, tipo)

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
    bod3 = fitpy10.fit3d(r_PH1, r_PH2, r_HH, dummy_grad)
    return bod3 + e12 + e13 + e23

# -----------------------------------------------------------------------------
# Spline H2 singlete (réplica Fortran)
# -----------------------------------------------------------------------------
_coeffs = None
_nodes  = None

def load_data(coeffs_path: str | Path, nodes_path:  str | Path) -> None:
    """Carga coeficientes (ncoeffs x 4) y nodos (nnodes)."""
    global _coeffs, _nodes
    coeffs_path = Path(coeffs_path)
    nodes_path  = Path(nodes_path)
    _coeffs = np.loadtxt(coeffs_path, dtype=np.float64)   # (ncoeffs, 4)
    _nodes  = np.loadtxt(nodes_path,  dtype=np.float64)   # (nnodes,)
    if _coeffs.ndim != 2 or _coeffs.shape[1] != 4:
        raise ValueError("coeffs debe ser (ncoeffs, 4)")
    if _nodes.ndim != 1 or _nodes.size != _coeffs.shape[0] + 1:
        raise ValueError("nnodes debe ser ncoeffs + 1")
    _coeffs = np.ascontiguousarray(_coeffs, dtype=np.float64)
    _nodes  = np.ascontiguousarray(_nodes,  dtype=np.float64)

def _ensure_loaded():
    if _coeffs is None or _nodes is None:
        raise RuntimeError("Debes llamar primero a load_data(coeffs_path, nodes_path)")

def _find_intervals(x: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(_nodes, x, side='right') - 1
    idx[(x < _nodes[0]) | (x > _nodes[-1])] = -1
    return idx

def spl_eval(x: Number) -> np.ndarray:
    """Evalúa el cúbico; fuera de rango → 0.0 (como tu Fortran)."""
    _ensure_loaded()
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.zeros_like(x_arr, dtype=np.float64)
    idx = _find_intervals(x_arr)
    valid = idx >= 0
    if np.any(valid):
        i = idx[valid]
        a, b, c, d = _coeffs[i, 0], _coeffs[i, 1], _coeffs[i, 2], _coeffs[i, 3]
        dx = x_arr[valid] - _nodes[i]
        y[valid] = a + b*dx + c*dx*dx + d*dx*dx*dx
    return y if isinstance(x, np.ndarray) else y.item()

def spl_deriv(x: Number) -> np.ndarray:
    """Derivada del cúbico; fuera de rango → 0.0."""
    _ensure_loaded()
    x_arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    dy = np.zeros_like(x_arr, dtype=np.float64)
    idx = _find_intervals(x_arr)
    valid = idx >= 0
    if np.any(valid):
        i = idx[valid]
        b, c, d = _coeffs[i, 1], _coeffs[i, 2], _coeffs[i, 3]
        dx = x_arr[valid] - _nodes[i]
        dy[valid] = b + 2.0*c*dx + 3.0*d*dx*dx
    return dy if isinstance(x, np.ndarray) else dy.item()

def diatHH_sing_py(r: Number) -> Tuple[np.ndarray, np.ndarray]:
    ener = spl_eval(r)
    der  = spl_deriv(r)
    return ener, der

# CARGA spline (ajusta rutas si es necesario)
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
# Gradiente y SD unificados (respetan GLOBAL_THETA si está activa)
# -----------------------------------------------------------------------------
def numerical_gradient(coords, tipo, h=1e-5):
    """
    - Sin θ: derivadas en r12, r13, r23 independientes (3D).
    - Con θ: solo r12 y r23 son independientes; r13 se reimpone (derivada tangente al colector).
    """
    xyz = np.array(coords, dtype=float)

    if GLOBAL_THETA is None:
        grad = np.zeros_like(xyz)
        for i in range(3):
            x1 = xyz.copy(); x2 = xyz.copy()
            x1[i] += h; x2[i] -= h
            f1 = PES_eval(x1, tipo); f2 = PES_eval(x2, tipo)
            grad[i] = (f1 - f2) / (2*h)
        return grad

    # θ fijo: variar r12 y r23; r13 proyectado cada vez
    r12, _, r23 = xyz
    f1 = PES_eval([r12+h, np.nan, r23], tipo)
    f2 = PES_eval([r12-h, np.nan, r23], tipo)
    g12 = (f1 - f2) / (2*h)

    f1 = PES_eval([r12, np.nan, r23+h], tipo)
    f2 = PES_eval([r12, np.nan, r23-h], tipo)
    g23 = (f1 - f2) / (2*h)

    return np.array([g12, 0.0, g23], dtype=float)

def steepest_descent_momentum(start_coords, max_steps, tipo,
                            step_size=0.05, tol=1e-10, momentum=0.85):
    coords = np.array(start_coords, dtype=float)
    if GLOBAL_THETA is not None:
        coords = project_to_theta(coords)
    path = [coords.copy()]
    prev = coords.copy()

    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
        grad = numerical_gradient(coords, tipo)
        gn = np.linalg.norm(grad[np.isfinite(grad)])
        if (not np.isfinite(gn)) or gn < tol:
            break
        update = -step_size * grad + momentum * (coords - prev)
        prev = coords.copy()
        coords = coords + update
        if GLOBAL_THETA is not None:
            coords = project_to_theta(coords)
        path.append(coords.copy())
    return np.array(path)

# -----------------------------------------------------------------------------
# ======= Puente con s = r23 - r12 constante (MISMA FIRMA) =======
#   - Si GLOBAL_THETA is None: minimiza en (r12, θ) con r23=r12+s.
#   - Si GLOBAL_THETA != None: impone θ fijo y minimiza solo en r12 (1D).
# -----------------------------------------------------------------------------
def build_bridge_constant_s(geom_A: np.ndarray, geom_B: np.ndarray, n_steps: int, tipo: str,
                            step=0.05, momentum=0.85, tol=1e-10, max_steps=2000,
                            r12_bounds=(0.5, 20.0)) -> np.ndarray:
    A = np.asarray(geom_A, dtype=float)
    B = np.asarray(geom_B, dtype=float)
    sA = A[2] - A[0]
    sB = B[2] - B[0]
    s_grid = np.linspace(sA, sB, int(n_steps)+1)

    path = []

    if GLOBAL_THETA is None:
        # ---------- Sin θ fija: minimizar en (r12, theta) ----------
        def _r13_from_theta_plus(r12: float, r23: float, theta: float) -> float:
            arg = r12*r12 + r23*r23 + 2.0*r12*r23*np.cos(theta)
            return np.sqrt(arg) if arg > 0.0 else np.inf

        def _theta_from_internals_plus(r12: float, r13: float, r23: float) -> float:
            c = (r13*r13 - r12*r12 - r23*r23) / (2.0*r12*r23)
            return np.arccos(np.clip(c, -1.0, 1.0))

        def _E_fixed_s(r12: float, s: float, theta: float, tipo: str) -> float:
            r23 = r12 + s
            if r12 <= 0.0 or r23 <= 0.0:
                return np.inf
            r13 = _r13_from_theta_plus(r12, r23, theta)
            if not np.isfinite(r13):
                return np.inf
            return PES((r12, r13, r23), tipo)

        def _grad_fixed_s(r12: float, s: float, theta: float, tipo: str,
                        h_r12=1e-4, h_th=1e-3) -> tuple[float, float]:
            f1 = _E_fixed_s(r12 + h_r12, s, theta, tipo)
            f2 = _E_fixed_s(r12 - h_r12, s, theta, tipo)
            g_r12 = (f1 - f2) / (2*h_r12)
            f1 = _E_fixed_s(r12, s, theta + h_th, tipo)
            f2 = _E_fixed_s(r12, s, theta - h_th, tipo)
            g_th = (f1 - f2) / (2*h_th)
            return g_r12, g_th

        r12_seed = A[0]
        th_seed  = _theta_from_internals_plus(A[0], A[1], A[2])
        lo, hi = r12_bounds

        for s in tqdm(s_grid, desc="Bridge SD: s-constant", ncols=100):
            r12, th = float(r12_seed), float(th_seed)
            v_r12, v_th = 0.0, 0.0
            for _ in range(max_steps):
                g_r12, g_th = _grad_fixed_s(r12, s, th, tipo)
                gn = np.hypot(g_r12, g_th)
                if not np.isfinite(gn) or gn < tol:
                    break
                v_r12 = momentum*v_r12 - step*g_r12
                v_th  = momentum*v_th  - step*g_th
                r12   = np.clip(r12 + v_r12, lo, hi)
                th    = np.clip(th + v_th, 0.0, np.pi)
            r23_opt = r12 + s
            r13_opt = _r13_from_theta_plus(r12, r23_opt, th)
            path.append([r12, r13_opt, r23_opt])
            r12_seed, th_seed = r12, th

    else:
        # ---------- Con θ fija: θ = GLOBAL_THETA, minimizar solo en r12 ----------
        theta_deg = float(GLOBAL_THETA)
        lo, hi = r12_bounds

        def E_fixed_s_theta(r12: float, s: float) -> float:
            r23 = r12 + s
            if r12 <= 0.0 or r23 <= 0.0:
                return np.inf
            r13 = r13_from_theta(r12, r23, theta_deg)
            if r13 is None:
                return np.inf
            return PES((r12, r13, r23), tipo)

        r12_seed = A[0]
        for s in tqdm(s_grid, desc=f"Bridge SD: s-constant (θ={theta_deg:g}°)", ncols=100):
            r12, v = float(r12_seed), 0.0
            for _ in range(max_steps):
                h = 1e-4
                f1 = E_fixed_s_theta(r12+h, s)
                f2 = E_fixed_s_theta(r12-h, s)
                g  = (f1 - f2) / (2*h)
                if not np.isfinite(g) or abs(g) < tol:
                    break
                v   = momentum*v - step*g
                r12 = np.clip(r12 + v, lo, hi)
            r23_opt = r12 + s
            r13_opt = r13_from_theta(r12, r23_opt, theta_deg)
            path.append([r12, r13_opt, r23_opt])
            r12_seed = r12

    return np.array(path, dtype=float)

# -----------------------------------------------------------------------------
# Interpolaciones (3D estándar y 2D con θ fijo)
# -----------------------------------------------------------------------------
def smooth_interpolation_on_pes(start_coords, end_coords, n_steps, tipo, energy_threshold=0.2):
    start = np.array(start_coords, dtype=float)
    end   = np.array(end_coords,   dtype=float)
    if GLOBAL_THETA is not None:
        start = project_to_theta(start); end = project_to_theta(end)

    path, energies = [], []
    for i in tqdm(range(n_steps + 1), desc="Interpolating", ncols=100, unit="step"):
        alpha = i / n_steps
        guess = (1 - alpha) * start + alpha * end
        if GLOBAL_THETA is not None:
            guess = project_to_theta(guess)
        E = PES_eval(guess, tipo)

        if i > 0 and abs(E - energies[-1]) > energy_threshold and np.isfinite(E):
            # mini corrección con gradiente tangente si hay θ
            grad = numerical_gradient(guess, tipo)
            gn = np.linalg.norm(grad[np.isfinite(grad)])
            if gn > 0:
                guess = guess - 0.05 * grad / gn
                if GLOBAL_THETA is not None:
                    guess = project_to_theta(guess)
                E = PES_eval(guess, tipo)

        path.append(guess)
        energies.append(E)

    return np.array(path), np.array(energies)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MEP (internas) con puente s-const y θ fijo opcional")
    parser.add_argument("--type", choices=["GS", "1QSm", "1TAp", "1SAp", "1SApp"], default="GS",
                        help="Tipo de superficie PES")
    parser.add_argument("--output", default="minimum_energy_path.dat",
                        help="Archivo de salida")
    parser.add_argument("--nbridge", type=int, default=10000,
                        help="Puntos en el puente (barrido en s)")
    parser.add_argument("--step", type=float, default=0.05,
                        help="Paso de SD (bridge y θ-modo)")
    parser.add_argument("--momentum", type=float, default=0.85,
                        help="Momentum de SD (bridge y θ-modo)")
    parser.add_argument("--maxsteps", type=int, default=2000,
                        help="Iteraciones internas de SD por punto s")
    parser.add_argument("--r12min", type=float, default=0.5)
    parser.add_argument("--r12max", type=float, default=20.0)
    parser.add_argument("--theta", type=float, default=None,
                        help="Ángulo fijo (grados). Si se omite, flujo 3D sin restricción.")
    args = parser.parse_args()
    tipo = args.type

    # <<< NUEVO: propaga θ global >>>
    global GLOBAL_THETA
    GLOBAL_THETA = args.theta  # None o valor en grados

    # Geometrías de inicio (internas r12, r13, r23)
    start1 = np.array([12.0, 12.0, 1.40], dtype=float)
    start2 = np.array([2.7, 12.0, 12.0], dtype=float)

    # ======================= Flujos por tipo =======================
    if tipo == "GS":
        path1 = steepest_descent_momentum(start1, 1_000_000, tipo)
        path2 = steepest_descent_momentum(start2,   100_000, tipo)[::-1]
        gA, gB = path1[-1], path2[0]
        bridge_path = build_bridge_constant_s(
            gA, gB, args.nbridge, tipo,
            step=args.step, momentum=args.momentum, max_steps=args.maxsteps,
            r12_bounds=(args.r12min, args.r12max)
        )
        final_path = np.vstack([path1, bridge_path, path2])

    elif tipo == "1QSm":
        geom_equil  = np.array([6.040695, 6.040695, 1.400000], dtype=float)
        geom_equil2 = np.array([2.80000000, 2.83139235, 3.16224475], dtype=float)
        path1, _ = smooth_interpolation_on_pes(start1, geom_equil, 1000, tipo)
        path2, _ = smooth_interpolation_on_pes(start2, geom_equil2, 1000, tipo)
        gA, gB = path1[-1], path2[0]
        bridge_path = build_bridge_constant_s(
            gA, gB, args.nbridge, tipo,
            step=args.step, momentum=args.momentum, max_steps=args.maxsteps,
            r12_bounds=(args.r12min, args.r12max)
        )
        final_path = np.vstack([path1, bridge_path, path2[::-1]])

    elif tipo == "1TAp":
        geom_equil  = np.array([6.040695, 6.040695, 1.400000], dtype=float)
        geom_equil2 = np.array([2.80000000, 2.83139235, 3.16224475], dtype=float)
        path1, _ = smooth_interpolation_on_pes(start1, geom_equil, 1000, tipo)
        path2, _ = smooth_interpolation_on_pes(start2, geom_equil2, 1000, tipo)
        gA, gB = path1[-1], path2[0]
        bridge_path = build_bridge_constant_s(
            gA, gB, args.nbridge, tipo,
            step=args.step, momentum=args.momentum, max_steps=args.maxsteps,
            r12_bounds=(args.r12min, args.r12max)
        )
        final_path = np.vstack([path1, bridge_path, path2[::-1]])

    elif tipo == "1SAp":
        path1 = steepest_descent_momentum(start1, 1_000_000, tipo)
        path2 = steepest_descent_momentum(start2,   100_000, tipo)[::-1]
        gA, gB = path1[-1], path2[0]
        bridge_path = build_bridge_constant_s(
            gA, gB, args.nbridge, tipo,
            step=args.step, momentum=args.momentum, max_steps=args.maxsteps,
            r12_bounds=(args.r12min, args.r12max)
        )
        final_path = np.vstack([path1, bridge_path, path2])

    elif tipo == "1SApp":
        geom_equil  = np.array([3.110048, 3.110048, 1.640000], dtype=float)
        geom_equil2 = np.array([2.70000000, 2.64611218, 4.81708436], dtype=float)
        path1, _ = smooth_interpolation_on_pes(start1, geom_equil, 1000, tipo)
        path2, _ = smooth_interpolation_on_pes(start2, geom_equil2, 1000, tipo)
        bridge_path, _ = smooth_interpolation_on_pes(geom_equil, geom_equil2, args.nbridge, tipo)
        final_path = np.vstack([path1, bridge_path, path2])

    else:
        raise ValueError(f"Tipo de PES no soportado: {tipo}")

    # Evaluar energía en el camino final (respetando θ si está activa)
    path_with_energy = np.column_stack([final_path, np.array([PES_eval(x, tipo) for x in final_path])])

    # -------------------------- Salida y gráfica --------------------------
    np.savetxt(args.output, path_with_energy[::2], header="r12 r13 r23 energy(Hartree)")
    print(f"Ruta guardada en {args.output}")

    E_all = path_with_energy[:, -1]
    i_min, i_max = np.argmin(E_all), np.argmax(E_all)
    print(f"Geom. mínima global: {path_with_energy[i_min, :3]}   Energía mínima: {E_all[i_min]:.10f}")
    print(f"Geom. máxima global: {path_with_energy[i_max, :3]}   Energía máxima: {E_all[i_max]:.10f}")

    plt.figure(figsize=(8, 5))
    plt.plot(E_all)
    plt.xlabel("Paso en el MEP")
    plt.ylabel("Energía (u.a.)")
    if len(E_all) >= 2:
        delta_E = (E_all[-1] - E_all[0]) * HARTREE_TO_EV
        plt.title(f"Evolución energética – tipo: {tipo}   ΔE = {delta_E:.2f} eV")
    else:
        plt.title(f"Evolución energética – tipo: {tipo}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
