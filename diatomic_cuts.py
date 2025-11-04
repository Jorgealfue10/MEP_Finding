#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrae curvas diatómicas como cortes 1D de la PES triatómica:
  - PH1: variar r_PH1, fijar r_PH2=L y r23=L
  - PH2: variar r_PH2, fijar r_PH1=L y r_HH=L
  - HH : variar r_HH,  fijar r_PH1=L y r_PH2=L

Opciones:
  - Desplazar a 0 la energía en el infinito (tres cuerpos separados) o al final del scan.
  - Comparar con potenciales diatómicos de tus librerías (diatphm, diathh).
  - Guardar .dat y figura.
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Tuple, Union

# ======================= Importa tus módulos existentes =======================
# Ajusta si tu ruta es distinta:
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

# ====================== Definición de la PES por tipo =========================

def energia1TApp(r12, r13, r23):
    e12, _ = diatphm(r12)
    e13, _ = diatphm(r13)
    e23, _ = diathh(r23)

    if (r12 < 2.0):
        e12=4.70442*np.exp(-0.948769*r12**2)-0.135445
    else:
        e12,_=diatphm(r12)
    if (r13 < 2.0):
        e13=4.70442*np.exp(-0.948769*r13**2)-0.135445
    else:
        e13,_=diatphm(r13)
    if (r23 < 0.74):
        e23=1.33889*r23**2 - 2.92997*r23 + 1.47035
    else:
        e23,_=diathh(r23)

    bod3, _ = fitGS.fit3d(r12, r13, r23)
    return bod3 + e12 + e13 + e23

def energia2TApp(r12, r13, r23):
    e12, _ = diat1qsm(r12)
    e13, _ = diat1qsm(r13)
    e23, _ = diathh(r23)

    if r12 < 2.0:
        e12 = 0.309099*r12**2.0-1.57688*r12+1.95195
    else:
        e12,_ = diat1qsm(r12)
    
    if r13 < 2.0:
        e13 = 0.309099*r13**2.0-1.57688*r13+1.95195
    else:
        e13,_ = diat1qsm(r13)

    if r23 < 0.74:
        e23 = 1.33889*r23**2.0-2.92997*r23+1.47035
    else:
        e23,_ = diathh(r23)

    bod3, _ = fit1QSm.fit3d(r12, r13, r23)
    return bod3 + e12 + e13 + e23

def energia1TAp(r12, r13, r23):
    e12, _ = diatphm(r12)
    e13, _ = diatphm(r13)
    e23, _ = diathh(r23)

    if (r12 < 2.0):
        e12=4.70442*np.exp(-0.948769*r12**2)-0.135445
    else:
        e12,_=diatphm(r12)
    if (r13 < 2.0):
        e13=4.70442*np.exp(-0.948769*r13**2)-0.135445
    else:
        e13,_=diatphm(r13)
    if (r23 < 0.74):
        e23=1.33889*r23**2 - 2.92997*r23 + 1.47035
    else:
        e23,_=diathh(r23)

    dummy_grad = [0.0, 0.0, 0.0]
    bod3 = fitpy8.fit3d(r12, r13, r23, dummy_grad)
    return bod3 + e12 + e13 + e23

# ---- spline H2 singlete para 1SAp / 1SApp (réplica Fortran) ----
_COEFFS = None
_NODES = None

def _load_spline(coeffs_path: str|Path, nodes_path: str|Path):
    global _COEFFS, _NODES
    _COEFFS = np.loadtxt(coeffs_path, dtype=float)
    _NODES  = np.loadtxt(nodes_path,  dtype=float)
    _COEFFS = np.ascontiguousarray(_COEFFS)
    _NODES  = np.ascontiguousarray(_NODES)

# Ajusta rutas si hace falta
_load_spline(
    "/home/jorgebdelafuente/Doctorado/RKHS/H2Psing-OK/cubic_spl_notrkhs/spline_coeffs.txt",
    "/home/jorgebdelafuente/Doctorado/RKHS/H2Psing-OK/cubic_spl_notrkhs/spline_nodes.txt",
)

def _find_int(x):
    i = np.searchsorted(_NODES, x, side='right') - 1
    i[(x < _NODES[0]) | (x > _NODES[-1])] = -1
    return i

def _spl_eval(x):
    x = np.atleast_1d(np.asarray(x, float))
    y = np.zeros_like(x)
    i = _find_int(x)
    ok = i >= 0
    if np.any(ok):
        a, b, c, d = _COEFFS[i[ok]].T
        dx = x[ok] - _NODES[i[ok]]
        y[ok] = a + b*dx + c*dx*dx + d*dx*dx*dx
    return y if y.ndim else float(y)

def _spl_deriv(x):
    x = np.atleast_1d(np.asarray(x, float))
    dy = np.zeros_like(x)
    i = _find_int(x)
    ok = i >= 0
    if np.any(ok):
        b, c, d = _COEFFS[i[ok], 1:].T
        dx = x[ok] - _NODES[i[ok]]
        dy[ok] = b + 2*c*dx + 3*d*dx*dx
    return dy if dy.ndim else float(dy)

def diatHH_sing_py(r):
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

def PES(coords, tipo="GS"):
    r12, r13, r23 = coords
    if   tipo == "GS":    return energia1TApp(r12, r13, r23)
    elif tipo == "1QSm":  return energia2TApp(r12, r13, r23)
    elif tipo == "1TAp":  return energia1TAp(r12, r13, r23)
    elif tipo == "1SAp":  return energia1SAp(r12, r13, r23)
    elif tipo == "1SApp": return energia1SApp(r12, r13, r23)
    else: raise ValueError(f"PES tipo desconocido: {tipo}")

# ======================= Utilidades de cortes diatómicos ======================

def make_rgrid(rmin: float, rmax: float, n: int) -> np.ndarray:
    """Malla 1D de distancias (incluye extremos)."""
    return np.linspace(float(rmin), float(rmax), int(n))

def diatomic_cut(tipo: str,
                 target: str,
                 r_vals: np.ndarray,
                 L: float = 25.0,
                 asymptote: str = "global") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae la curva diatómica como corte 1D de la PES.
      - tipo: "GS", "1QSm", "1TAp", "1SAp", "1SApp"
      - target: "PH1", "PH2" o "HH"
      - r_vals: malla 1D (a0) para la distancia variable
      - L: distancia grande para los otros dos enlaces (a0)
      - asymptote:
          "none"    -> no desplaza energía
          "global"  -> resta PES(L,L,L)
          "variable"-> resta energía en el último punto del scan, con los otros dos en L

    Devuelve (r_vals, E, E_shifted) en Hartree.
    """
    target = target.upper()
    E = np.zeros_like(r_vals, dtype=float)
    L = float(L)

    for i, r in enumerate(r_vals):
        if target == "PH1":
            coords = np.array([r, L, L], dtype=float)
        elif target == "PH2":
            coords = np.array([L, r, L], dtype=float)
        elif target in {"HH", "H-H", "H2"}:
            coords = np.array([L, L, r], dtype=float)
        else:
            raise ValueError("target debe ser 'PH1', 'PH2' o 'HH'.")
        E[i] = PES(coords, tipo)

    if asymptote == "none":
        E_shift = E.copy()
    elif asymptote == "global":
        E_inf = PES(np.array([L, L, L], dtype=float), tipo)
        E_shift = E - E_inf
    elif asymptote == "variable":
        # Resta la energía del último punto del scan (r=r_max) con los otros dos a L
        if target == "PH1":
            E_inf = PES(np.array([r_vals[-1], L, L], dtype=float), tipo)
        elif target == "PH2":
            E_inf = PES(np.array([L, r_vals[-1], L], dtype=float), tipo)
        else:  # HH
            E_inf = PES(np.array([L, L, r_vals[-1]], dtype=float), tipo)
        E_shift = E - E_inf
    else:
        raise ValueError("asymptote ∈ {'none','global','variable'}")

    return r_vals, E, E_shift

def compare_with_diatomics_library(r_vals: np.ndarray, target: str) -> np.ndarray:
    """
    (Opcional) Devuelve la diatómica 'pura' para comparar:
      - PH: diatphm
      - HH: diathh
    """
    target = target.upper()
    V = np.zeros_like(r_vals, dtype=float)
    if target in {"PH1", "PH2"}:
        for i, r in enumerate(r_vals):
            V[i], _ = diatphm(r)
    elif target in {"HH", "H-H", "H2"}:
        for i, r in enumerate(r_vals):
            V[i], _ = diathh(r)
    else:
        raise ValueError("target debe ser 'PH1', 'PH2' o 'HH'.")
    return V

# ================================== CLI ======================================

def main():
    ap = argparse.ArgumentParser(description="Cortes diatómicos desde la PES triatómica")
    ap.add_argument("--type", choices=["GS","1QSm","1TAp","1SAp","1SApp"], default="GS",
                    help="Tipo de superficie PES")
    ap.add_argument("--target", choices=["PH1","PH2","HH"], default="PH1",
                    help="Diátomo a extraer")
    ap.add_argument("--rmin", type=float, default=0.6, help="Distancia mínima (a0)")
    ap.add_argument("--rmax", type=float, default=10.0, help="Distancia máxima (a0)")
    ap.add_argument("--n", type=int, default=400, help="Número de puntos (incluye extremos)")
    ap.add_argument("--L", type=float, default=25.0, help="Distancia grande para enlaces espectadores (a0)")
    ap.add_argument("--asymptote", choices=["none","global","variable"], default="global",
                    help="Cómo desplazar la energía a 0 en el infinito")
    ap.add_argument("--compare", action="store_true", help="Comparar con potencial diatómico de librería")
    ap.add_argument("--out", type=str, default=None, help="Archivo .dat de salida (auto si no se indica)")
    ap.add_argument("--plot", type=str, default=None, help="Guardar figura (png/pdf). Si se omite, solo muestra.")
    args = ap.parse_args()

    r_vals = make_rgrid(args.rmin, args.rmax, args.n)
    r, E, E0 = diatomic_cut(tipo=args.type, target=args.target, r_vals=r_vals,
                            L=args.L, asymptote=args.asymptote)

    minval = min(E)
    minval0 = min(E0)
    E0 -= minval0
    E -= minval

    # Guardado .dat
    if args.out is None:
        args.out = f"diatomic_{args.target}_{args.type}.dat"
    data = np.column_stack([r, E, E0])
    header = f"r(a0)  E(Hartree)  E_shifted(Hartree)  # target={args.target} type={args.type} L={args.L} asymptote={args.asymptote}"
    np.savetxt(args.out, data, header=header)
    print(f"[OK] Guardado: {args.out}")

    # Plot
    plt.figure(figsize=(7.0, 4.6))
    plt.plot(r, E0, label=f"Corte PES ({args.target}) – desplazado")
    plt.plot(r, E,  alpha=0.5, linestyle=":", label="Corte PES (sin desplazar)")
    if args.compare:
        V = compare_with_diatomics_library(r, args.target)
        plt.plot(r, V, linestyle="--", label="Diátomo (librería)")
    plt.xlabel("r (a$_0$)")
    plt.ylabel("Energía (u.a.)")
    plt.title(f"{args.target} en {args.type}  (L={args.L} a$_0$, asy='{args.asymptote}')")
    plt.grid(True); plt.legend(); plt.tight_layout()

    if args.plot:
        plt.savefig(args.plot, dpi=200, bbox_inches="tight")
        print(f"[OK] Figura guardada: {args.plot}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
