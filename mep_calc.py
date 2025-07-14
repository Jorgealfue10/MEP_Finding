import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Añadir rutas a tus módulos personalizados
sys.path.append('/home/jorgebdelafuente/Codes/MEP_Finding/')
from PHMGS import *
from PHM1QSm import *
from H2_diat import *
import fitGS
import fit1QSm
import fitpy8

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
def steepest_descent_momentum(start_coords, max_steps, tipo, step_size=0.25, tol=1e-10, momentum=1.0):
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
    parser.add_argument("--type", choices=["GS", "1QSm", "1TAp"], default="GS",
                        help="Tipo de superficie PES: 'GS', '1QSm' o '1TAp'")
    parser.add_argument("--output", default="minimum_energy_path.dat",
                        help="Nombre del archivo de salida")
    args = parser.parse_args()
    tipo = args.type

    print(f"Usando superficie PES tipo: {tipo}")

    start1 = np.array([15.0, 15.0, 1.40])
    start2 = np.array([2.7, 15.0, 15.0])
    

    # Calcular caminos
    if tipo == "GS":
        path1 = steepest_descent_momentum(start1, 100000, tipo)
        path2 = steepest_descent_momentum(start2, 100000, tipo)[::-1]
        path_middle, _ = smooth_interpolation_on_pes(path1[-1], path2[0], 10000, tipo)
        final_path = np.vstack([path1, path_middle, path2])
    elif tipo == "1QSm":
        geom_equil = np.array([6.040695, 6.040695, 1.400000])  
        geom_equil2 = np.array([2.80000000, 2.83139235, 3.16224475])  
        path1, _ = smooth_interpolation_on_pes(start1, geom_equil, 1000, tipo)
        path2, _ = smooth_interpolation_on_pes(start2, geom_equil2, 1000, tipo)
        bridge_path, _ = smooth_interpolation_on_pes(geom_equil,geom_equil2, 10000, tipo)
        final_path = np.vstack([path1, bridge_path, path2[::-1]])  
    elif tipo == "1TAp":
        geom_equil = np.array([6.040695, 6.040695, 1.400000])  
        geom_equil2 = np.array([2.80000000, 2.83139235, 3.16224475])  
        path1, _ = smooth_interpolation_on_pes(start1, geom_equil, 1000, tipo)
        path2, _ = smooth_interpolation_on_pes(start2, geom_equil2, 1000, tipo)
        bridge_path, _ = smooth_interpolation_on_pes(geom_equil, geom_equil2, 10000, tipo)
        final_path = np.vstack([path1, bridge_path, path2[::-1]])


    # Evaluar energía
    path_with_energy = []
    for coords in final_path:
        energy = PES(coords, tipo)
        path_with_energy.append(np.append(coords, energy))
    path_with_energy = np.array(path_with_energy)

    # Guardar resultados
    np.savetxt(args.output, path_with_energy[::2], header="r1 r2 r3 energy")
    print(f"Ruta guardada en {args.output}")

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
