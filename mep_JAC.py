#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from pathlib import Path
from typing import Tuple, Union
from matplotlib.colors import Normalize

# ----------------------- Importa tus módulos existentes -----------------------
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
HARTREE_TO_EV = 27.2114

# ============================= PES EN INTERNAS =================================
def energia1TApp(r12, r13, r23):
    e12, _ = diatphm(r12)
    e13, _ = diatphm(r13)
    e23, _ = diathh(r23)
    bod3, _ = fitGS.fit3d(r12, r13, r23)
    return bod3 + e12 + e13 + e23

def energia2TApp(r12, r13, r23):
    e12, _ = diat1qsm(r12)
    e13, _ = diat1qsm(r13)
    e23, _ = diathh(r23)
    bod3, _ = fit1QSm.fit3d(r12, r13, r23)
    return bod3 + e12 + e13 + e23

def energia1TAp(r12, r13, r23):
    e12, _ = diatphm(r12)
    e13, _ = diatphm(r13)
    e23, _ = diathh(r23)
    dummy_grad = [0.0, 0.0, 0.0]
    bod3 = fitpy8.fit3d(r12, r13, r23, dummy_grad)
    return bod3 + e12 + e13 + e23

# ---- spline H2 singlete (tu réplica Fortran) ----
# (ajusta rutas si hace falta)
_COEFFS = None
_NODES = None

def _load_spline(coeffs_path: str|Path, nodes_path: str|Path):
    global _COEFFS, _NODES
    _COEFFS = np.loadtxt(coeffs_path, dtype=float)
    _NODES  = np.loadtxt(nodes_path,  dtype=float)
    _COEFFS = np.ascontiguousarray(_COEFFS)
    _NODES  = np.ascontiguousarray(_NODES)

def _find_int(x):
    i = np.searchsorted(_NODES, x, side='right')-1
    i[(x < _NODES[0]) | (x > _NODES[-1])] = -1
    return i

def _spl_eval(x):
    x = np.atleast_1d(np.asarray(x, float))
    y = np.zeros_like(x)
    i = _find_int(x)
    ok = i >= 0
    if np.any(ok):
        a,b,c,d = _COEFFS[i[ok]].T
        dx = x[ok] - _NODES[i[ok]]
        y[ok] = a + b*dx + c*dx*dx + d*dx*dx*dx
    return y if y.ndim else y.item()

def _spl_deriv(x):
    x = np.atleast_1d(np.asarray(x, float))
    dy = np.zeros_like(x)
    i = _find_int(x)
    ok = i >= 0
    if np.any(ok):
        b,c,d = _COEFFS[i[ok],1:].T
        dx = x[ok] - _NODES[i[ok]]
        dy[ok] = b + 2*c*dx + 3*d*dx*dx
    return dy if dy.ndim else dy.item()

# Carga por defecto (ajusta rutas si hiciera falta)
_load_spline(
    "/home/jorgebdelafuente/Doctorado/RKHS/H2Psing-OK/cubic_spl_notrkhs/spline_coeffs.txt",
    "/home/jorgebdelafuente/Doctorado/RKHS/H2Psing-OK/cubic_spl_notrkhs/spline_nodes.txt",
)

def diatHH_sing_py(r):  # firma compatible
    return _spl_eval(r), _spl_deriv(r)

def energia1SAp(r12, r13, r23):
    e12, _ = diatphm(r12)
    e13, _ = diatphm(r13)
    e23, _ = diatHH_sing_py(r23)
    bod3, _ = SAp10.fit3d(r12, r13, r23)
    return bod3 + e12 + e13 + e23

def energia1SApp(r12, r13, r23):
    e12, _ = diatphm(r12)
    e13, _ = diatphm(r13)
    e23, _ = diatHH_sing_py(r23)
    bod3, _ = SApp10.fit3d(r12, r13, r23)
    return bod3 + e12 + e13 + e23

# Selector de PES en internucleares
def PES_internals(r12, r13, r23, tipo="GS"):
    if   tipo == "GS":     return energia1TApp(r12, r13, r23)
    elif tipo == "1QSm":   return energia2TApp(r12, r13, r23)
    elif tipo == "1TAp":   return energia1TAp(r12, r13, r23)
    elif tipo == "1SAp":   return energia1SAp(r12, r13, r23)
    elif tipo == "1SApp":  return energia1SApp(r12, r13, r23)
    else:
        raise ValueError(f"PES tipo desconocido: {tipo}")

# =========================== JACOBI ↔ INTERNAS ================================
# 1–{2,3}  (átomo 1 y diátomo 2-3)
def internals_from_jacobi(R: float, r: float, theta_deg: float,
                        m2: float, m3: float) -> Tuple[float,float,float]:
    """Devuelve (r12,r13,r23) a partir de (R,r,theta)."""
    theta = np.radians(theta_deg)
    M23 = m2 + m3
    mu2 = m2 / M23
    mu3 = m3 / M23
    r23 = r
    r12 = np.sqrt(R*R + (mu3*r)**2 - 2.0*mu3*R*r*np.cos(theta))
    r13 = np.sqrt(R*R + (mu2*r)**2 + 2.0*mu2*R*r*np.cos(theta))
    return r12, r13, r23

# ============================= PES EN JACOBI ==================================
def PES_jacobi(R: float, r: float, theta_deg: float, tipo: str,
            m2: float, m3: float) -> float:
    r12, r13, r23 = internals_from_jacobi(R, r, theta_deg, m2, m3)
    return PES_internals(r12, r13, r23, tipo)

# ============================== GRADIENTES ====================================
def numerical_gradient_jacobi(X, tipo, m2, m3, h=(1e-4, 1e-4, 1e-3)):
    """Gradiente numérico ∂E/∂(R,r,theta_deg). Usa diferencias centradas."""
    R, r, th = X
    hR, hr, hth = h
    # R
    e1 = PES_jacobi(R+hR, r, th, tipo, m2, m3)
    e2 = PES_jacobi(R-hR, r, th, tipo, m2, m3)
    dR = (e1-e2)/(2*hR)
    # r
    e1 = PES_jacobi(R, r+hr, th, tipo, m2, m3)
    e2 = PES_jacobi(R, r-hr, th, tipo, m2, m3)
    dr = (e1-e2)/(2*hr)
    # theta (en grados)
    e1 = PES_jacobi(R, r, th+hth, tipo, m2, m3)
    e2 = PES_jacobi(R, r, th-hth, tipo, m2, m3)
    dth = (e1-e2)/(2*hth)
    return np.array([dR, dr, dth])

def numerical_gradient_jacobi_fixed_theta(Y, tipo, theta_deg, m2, m3, h=(1e-4, 1e-4)):
    """Gradiente en (R,r) con θ fijo."""
    R, r = Y
    hR, hr = h
    e1 = PES_jacobi(R+hR, r, theta_deg, tipo, m2, m3)
    e2 = PES_jacobi(R-hR, r, theta_deg, tipo, m2, m3)
    dR = (e1-e2)/(2*hR)
    e1 = PES_jacobi(R, r+hr, theta_deg, tipo, m2, m3)
    e2 = PES_jacobi(R, r-hr, theta_deg, tipo, m2, m3)
    dr = (e1-e2)/(2*hr)
    return np.array([dR, dr])

# ======================== OPTIMIZACIÓN/INTERPOLACIÓN ==========================
def steepest_descent_momentum(fgrad, x0, max_steps=200000, step=0.05, tol=1e-10, mom=0.85):
    path = [np.array(x0, dtype=float)]
    x = np.array(x0, dtype=float)
    prev = x.copy()
    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
        g = fgrad(x)
        gn = np.linalg.norm(g)
        if not np.isfinite(gn) or gn < tol:
            break
        upd = -step*g + mom*(x - prev)
        prev = x.copy()
        x = x + upd
        path.append(x.copy())
    return np.array(path)

def smooth_interpolation(f, xA, xB, n, energy_threshold=0.2):
    xA = np.array(xA, float)
    xB = np.array(xB, float)
    path, Ener = [], []
    for i in tqdm(range(n+1), desc="Interpolating", ncols=100, unit="step"):
        a = i/n
        g = (1-a)*xA + a*xB
        E = f(g)
        if i>0 and abs(E - Ener[-1]) > energy_threshold:
            # Un paso de corrección "suave" hacia menor energía
            # (dirección del gradiente numérico via diferencias unilaterales)
            eps = 1e-4
            d = np.zeros_like(g)
            for k in range(len(g)):
                g2 = g.copy(); g2[k] += eps
                d[k] = (f(g2) - E)/eps
            dn = np.linalg.norm(d)
            if dn > 0 and np.isfinite(dn):
                g = g - 0.05*d/dn
                E = f(g)
        path.append(g)
        Ener.append(E)
    return np.array(path), np.array(Ener)

# ============================== PLOT DE CONTORNOS =============================
def rounded_levels(vmin, vmax, step=0.5):
    """Niveles redondeados (eV) que incluyen 0."""
    # redondeo a múltiplos de 'step'
    vmin = step * np.floor(vmin/step)
    vmax = step * np.ceil(vmax/step)
    lv = np.arange(vmin, vmax + step, step)
    if 0.0 not in lv:
        lv = np.sort(np.append(lv, 0.0))
    return lv

def contour_R_r(theta_deg, tipo, m2, m3,
                R_range=(1.5, 15.0), r_range=(0.6, 15.0),
                nR=220, nr=220, vmin_eV=-7.0, vmax_eV=1.0, step_eV=0.5):
    R = np.linspace(*R_range, nR)
    r = np.linspace(*r_range, nr)
    RR, rr = np.meshgrid(R, r, indexing='xy')
    E = np.zeros_like(RR)

    for i in tqdm(range(nR), desc=f"E(R,r) θ={theta_deg:g}°", ncols=100):
        for j in range(nr):
            E[j,i] = PES_jacobi(R[i], r[j], theta_deg, tipo, m2, m3)

    # a eV para colorear
    E_eV = E * HARTREE_TO_EV
    levels = rounded_levels(vmin_eV, vmax_eV, step=step_eV)
    cmap = plt.get_cmap('gist_ncar').copy()
    norm = Normalize(vmin=vmin_eV, vmax=vmax_eV, clip=False)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    cf = ax.contourf(RR, rr, E_eV, levels=levels, cmap=cmap, norm=norm, extend='both')
    ax.contour(RR, rr, E_eV, levels=[0.0], colors='black', linewidths=1.2)  # línea 0 eV

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Energy (eV)")
    cbar.set_ticks(levels)

    ax.set_xlabel(r"$R_{1-(23)}$ (a$_0$)")
    ax.set_ylabel(r"$r_{23}$ (a$_0$)")
    ax.set_title(f"{tipo}  –  $\\theta$ = {theta_deg:.0f}°")
    plt.tight_layout()
    plt.show()

# ================================== MAIN =====================================
def main():
    p = argparse.ArgumentParser(description="MEP y contornos en coordenadas de Jacobi (1–{2,3})")
    p.add_argument("--type", choices=["GS","1QSm","1TAp","1SAp","1SApp"], default="GS")
    p.add_argument("--theta", type=float, default=None,
                help="Ángulo θ (grados). Si lo das, se usa modo 2D (R,r) a θ fijo.")
    p.add_argument("--m2", type=float, default=1.0, help="Masa atómica del átomo 2 (u).")
    p.add_argument("--m3", type=float, default=1.0, help="Masa atómica del átomo 3 (u).")
    p.add_argument("--contours", action="store_true", help="Dibujar contornos (R,r) a θ fijo.")
    p.add_argument("--Rmin", type=float, default=2.0)
    p.add_argument("--Rmax", type=float, default=15.0)
    p.add_argument("--rmin", type=float, default=0.6)
    p.add_argument("--rmax", type=float, default=15.0)
    args = p.parse_args()

    tipo = args.type
    m2, m3 = args.m2, args.m3

    # ----------------- Solo contornos a θ fijo (rápido de usar) ----------------
    if args.contours:
        if args.theta is None:
            raise SystemExit("Necesitas --theta para dibujar contornos (R,r).")
        contour_R_r(
            theta_deg=args.theta, tipo=tipo, m2=m2, m3=m3,
            R_range=(args.Rmin, args.Rmax), r_range=(args.rmin, args.rmax),
            vmin_eV=-7.0, vmax_eV=1.0, step_eV=0.5
        )
        return

    # ---------------------- Ejemplo de MEP a θ fijo ----------------------------
    if args.theta is not None:
        theta = args.theta
        # funciones de ayuda con θ fijo
        fE = lambda Y: PES_jacobi(Y[0], Y[1], theta, tipo, m2, m3)
        fG = lambda Y: numerical_gradient_jacobi_fixed_theta(Y, tipo, theta, m2, m3)

        # puntos de inicio típicos (ajusta a tu sistema)
        start_in  = np.array([12.0, 1.40])  # R grande, H2 cercano
        start_out = np.array([2.7, 12.0])   # R pequeño, r grande

        path1 = steepest_descent_momentum(fG, start_in,  max_steps=1_000_000)
        path2 = steepest_descent_momentum(fG, start_out, max_steps=100_000)[::-1]
        bridge, _ = smooth_interpolation(fE, path1[-1], path2[0], 100_000)
        final2d = np.vstack([path1, bridge, path2])

        # reconstruye (r12,r13,r23) y energía
        out = []
        for R, r in final2d:
            r12, r13, r23 = internals_from_jacobi(R, r, theta, m2, m3)
            E = PES_internals(r12, r13, r23, tipo)
            out.append([R, r, theta, r12, r13, r23, E])
        out = np.array(out)

        np.savetxt("mep_jacobi_theta.dat", out[::2],
                header="R r theta_deg r12 r13 r23 Energy(Hartree)")
        print("Ruta (Jacobi, θ fijo) guardada en mep_jacobi_theta.dat")

        # gráfico energía
        plt.figure(figsize=(8,5))
        plt.plot(out[:, -1], color='black')
        plt.xlabel("Paso en el MEP (θ fijo)")
        plt.ylabel("Energía (u.a.)")
        dE = (out[-1,-1] - out[0,-1]) * HARTREE_TO_EV
        plt.title(f"MEP en Jacobi – {tipo} – θ={theta:.0f}°  (ΔE={dE:.2f} eV)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return

    # ------------------------- MEP 3D (R, r, theta) ---------------------------
    fE3 = lambda X: PES_jacobi(X[0], X[1], X[2], tipo, m2, m3)
    fG3 = lambda X: numerical_gradient_jacobi(X, tipo, m2, m3)

    # ejemplos de arranque (ajusta a tu caso)
    xA = np.array([12.0, 1.40, 0.0])   # R grande, r ~ eq, θ=0
    xB = np.array([2.7, 12.0, 180.0])  # R pequeño, r grande, θ=180

    pathA = steepest_descent_momentum(fG3, xA, 1_000_000)
    pathB = steepest_descent_momentum(fG3, xB, 100_000)[::-1]
    bridge3, _ = smooth_interpolation(fE3, pathA[-1], pathB[0], 100_000)
    final3 = np.vstack([pathA, bridge3, pathB])

    # guarda
    Ener = np.array([fE3(p) for p in final3])
    out3 = np.column_stack([final3, Ener])
    np.savetxt("mep_jacobi_3d.dat", out3[::2],
            header="R r theta_deg Energy(Hartree)")
    print("Ruta (Jacobi 3D) guardada en mep_jacobi_3d.dat")

    # plot
    plt.figure(figsize=(8,5))
    plt.plot(Ener, color='black')
    plt.xlabel("Paso en el MEP (3D)")
    plt.ylabel("Energía (u.a.)")
    dE = (Ener[-1] - Ener[0]) * HARTREE_TO_EV
    plt.title(f"MEP 3D en Jacobi – {tipo}  (ΔE={dE:.2f} eV)")
    plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
