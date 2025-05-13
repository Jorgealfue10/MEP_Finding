import numpy as np
from scipy.optimize import minimize
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/home/jorgebdelafuente/Codes/MEP_Finding/')
from PHMGS import *
from PHM1QSm import *
from H2_diat import *
import fitGS 
import fit1QSm


# ─────────────────────────────────────────────────────────────────────────────
# PES CON SELECTOR DE TIPO
# ─────────────────────────────────────────────────────────────────────────────
def PES(coords, tipo):
    r1, r2, r3 = coords
    der = np.zeros(3)

    if tipo == "GS":
        diat12, d12 = diatphm(r1)
        diat13, d13 = diatphm(r2)
        diat23, d23 = diathh(r3)
        t3bod, d3 = fitGS.fit3d(r1, r2, r3)
    elif tipo == "1QSm":
        if r1 < 2.0:
            diat12 = 0.309099 * r1**2.0 - 1.57688 * r1 + 1.95195
            d12 = 0.309099 * 2 * r1 - 1.57688
        else:
            diat12, d12 = diat1qsm(r1)

        if r2 < 2.0:
            diat13 = 0.309099 * r2**2.0 - 1.57688 * r2 + 1.95195
            d13 = 0.309099 * 2 * r2 - 1.57688
        else:
            diat13, d13 = diat1qsm(r2)

        if r3 < 0.74:
            diat23 = 1.33889 * r3**2.0 - 2.92997 * r3 + 1.47035
            d23 = 1.33889 * 2 * r3 - 2.92997
        else:
            diat23, d23 = diathh(r3)
        t3bod, d3 = fit1QSm.fit3d(r1, r2, r3)
    else:
        raise ValueError(f"Tipo de PES no reconocido: '{tipo}'")

    der[0] = d12 + d3[0]
    der[1] = d13 + d3[1]
    der[2] = d23 + d3[2]

    return diat12 + diat13 + diat23 + t3bod, der

# ─────────────────────────────────────────────────────────────────────────────

# Método de Steepest Descent
def steepest_descent(start_coords, step_size=0.1, tol=1e-10, max_steps=100000):
    path = [start_coords]
    coords = np.array(start_coords)

    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
    # for _ in range(max_steps):
        _,grad = PES(coords,tipo=PES_TYPE)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break  # Nos detenemos si el gradiente es muy pequeño
        
        # Paso en dirección de mayor descenso
        coords = coords - step_size * grad / grad_norm
        path.append(coords)

    return np.array(path)

def steepest_crescent_smooth(start_coords, step_size=0.1, tol=1e-10, max_steps=100000):
    path = [start_coords]
    coords = np.array(start_coords)

    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
    # for _ in range(max_steps):
        energy, grad = PES(coords,tipo=PES_TYPE)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break  # Nos detenemos si el gradiente es muy pequeño
        
        # Encontrar la dirección con el menor valor absoluto del gradiente
        grad_sign = np.sign(grad)
        grad_abs = np.abs(grad)
        smallest_grad_index = np.argmin(grad_abs)

        # Avanzar en la dirección de menor gradiente positivo
        step_direction = np.zeros_like(grad)
        step_direction[smallest_grad_index] = grad[smallest_grad_index]

        coords = coords + step_size * step_direction / np.linalg.norm(step_direction)
        path.append(coords)

    return np.array(path)

############################################################################################

# Método de Steepest Descent con Atracción
def steepest_descent_with_attraction(start_coords, end_coords, alpha=0.1, beta=0.75, tol=1e-3, max_steps=100000):
    """
    Conecta los dos mínimos con Steepest Descent y una fuerza de atracción hacia el mínimo final.
    
    start_coords: Coordenadas iniciales del primer mínimo.
    end_coords: Coordenadas del segundo mínimo.
    alpha: Tamaño de paso para el gradiente descendente.
    beta: Fuerza de atracción hacia el mínimo final.
    tol: Tolerancia para el gradiente.
    max_steps: Número máximo de pasos.
    """
    path = [start_coords]
    coords = np.array(start_coords)

    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
    # for _ in range(max_steps):
        # Obtener el gradiente de la PES
        _, grad = PES(coords,tipo=PES_TYPE)
        grad_norm = np.linalg.norm(grad)

        # Si el gradiente es suficientemente pequeño, detenemos el proceso
        if (start_coords-end_coords).all() < tol:
            break
        
        # Dirección de atracción hacia el otro mínimo
        attraction = end_coords - coords
        attraction_norm = np.linalg.norm(attraction)

        # Normalizamos la atracción para que sea una dirección unitaria
        if attraction_norm > 0:
            attraction /= attraction_norm
        
        # Paso de Steepest Descent más atracción
        grad_direction = grad / grad_norm  # Dirección del gradiente
        coords = coords - alpha * grad_direction + beta * attraction

        # Añadir las nuevas coordenadas al camino
        path.append(coords)

    return np.array(path)

############################################################################################

# def steepest_descent_momentum(start_coords, step_size=0.1, tol=1e-10, max_steps=1000000, momentum=0.75):
def steepest_descent_momentum(start_coords, max_steps, step_size=0.2, tol=1e-10, momentum=0.75):
    path = [start_coords]
    coords = np.array(start_coords)
    prev_coords = np.array(start_coords)  # La posición anterior (inicialmente igual a start_coords)
    prev_grad = np.zeros_like(coords)  # Gradiente del paso anterior

    # for _ in range(max_steps):
    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
        energy, grad = PES(coords,tipo=PES_TYPE)  # Obtener la energía y el gradiente
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break  # Nos detenemos si el gradiente es muy pequeño
        
        # Aplicar la fórmula con momento
        update = -step_size * grad + momentum * (coords - prev_coords)
        
        # Actualizar las coordenadas
        prev_coords = np.copy(coords)  # Guardamos la posición actual como la anterior para el próximo paso
        prev_grad = np.copy(grad)  # Guardamos el gradiente actual como el anterior para el próximo paso
        coords = coords + update  # Actualizar la posición

        path.append(coords)

    return np.array(path)

############################################################################################

def smooth_interpolation_on_pes(start_coords, end_coords, n_steps, energy_threshold=0.2):
    """
    Interpola geométricamente entre dos estructuras, evaluando en la PES para asegurar
    que no hay saltos energéticos fuertes.
    """
    start = np.array(start_coords)
    end = np.array(end_coords)
    
    path = []
    energies = []

    for i in tqdm(range(n_steps + 1), desc="Interpolating", ncols=100, unit="step"):
        # Interpolación lineal
        alpha = i / n_steps
        guess = (1 - alpha) * start + alpha * end

        # Evaluar energía en el punto interpolado
        E, grad = PES(guess,tipo=PES_TYPE)

        if i > 0 and abs(E - energies[-1]) > energy_threshold:
            # Suavizamos si hay un salto: damos un pasito en dirección de menor energía
            # pero sin buscar un mínimo (solo una corrección)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 0:
                guess = guess - 0.05 * grad / grad_norm
                E, grad = PES(guess)

        path.append(guess)
        energies.append(E)

    return np.array(path), np.array(energies)

############################################################################################

def main():
    global PES_TYPE

    # Argumentos desde terminal
    parser = argparse.ArgumentParser(description="MEP optimization script")
    parser.add_argument("--type", choices=["GS", "1QSm"], default="GS",
                        help="Tipo de superficie PES: 'GS' o '1QSm'")
    parser.add_argument("--output", default="minimum_energy_path.dat",
                        help="Nombre del archivo de salida")
    args = parser.parse_args()
    PES_TYPE = args.type

    print(f"Usando superficie PES tipo: {PES_TYPE}")

    # Límites iniciales comunes
    start1 = np.array([20., 20., 1.40])
    start2 = np.array([2.7, 20., 20.5])

    if PES_TYPE == "GS":
        # Caminos desde extremos con descenso
        path1 = steepest_descent_momentum(start1, 1000000)
        path2 = steepest_descent_momentum(start2, 1000000)[::-1]

        # Interpolación suave entre extremos
        bridge_path, _ = smooth_interpolation_on_pes(path1[-1], path2[0], 1000)

        final_path = np.vstack([path1, bridge_path, path2])

        # Evaluar energía
        path_with_energy = []
        for coords in final_path:
            energy, _ = PES(coords,tipo=PES_TYPE)
            path_with_energy.append(np.append(coords, energy))
        path_with_energy = np.array(path_with_energy)

    elif PES_TYPE == "1QSm":
        path1 = steepest_descent_momentum(start1, 1000000)
        path2 = steepest_descent_momentum(start2, 1000000)[::-1]
        # Interpolación directa sin descenso
        final_path, energies = smooth_interpolation_on_pes(path1[-1], path2[0], n_steps=1000000)
        path_with_energy = np.hstack([final_path, energies.reshape(-1, 1)])

    else:
        raise ValueError(f"Tipo de PES no soportado: {PES_TYPE}")

    # Guardar
    np.savetxt(args.output, path_with_energy[::2], header="r1 r2 r3 energy")
    print(f"Ruta guardada en {args.output}")

    # Graficar energía
    r1, r2, r3, energy = path_with_energy.T
    plt.figure(figsize=(8, 5))
    plt.plot(energy, color="black")
    plt.xlabel("Paso en el MEP")
    plt.ylabel("Energía (u.a.)")
    plt.title(f"Evolución energética – tipo: {PES_TYPE}")
    delta_E = (energy[-1] - energy[0]) * 27.2114
    plt.annotate(f"$\\Delta E$ = {delta_E:.2f} eV", xy=(0.5, 0.95), xycoords="axes fraction",
                 ha="center", va="top", size=12)
    plt.grid()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()