import numpy as np
from scipy.optimize import minimize
from diatPHM import *
from diatH2 import *
from body3 import *
import matplotlib.pyplot as plt
from tqdm import tqdm

# Función de la PES (debes definirla con tu expresión analítica)
def PES(coords):
    r1, r2, r3 = coords  # Coordenadas internas (distancias interatómicas)
    diat12,d12=diat12doub1ap(r1)
    diat13,d13=diat12doub1ap(r2)
    diat23,d23=diathh(r3)
    t3bod,der=fit3d(r1,r2,r3)
    der[0]=d12+der[0]
    der[1]=d13+der[1]
    der[2]=d23+der[2]
    return diat12+diat13+diat23+t3bod,der

# Método de Steepest Descent
def steepest_descent(start_coords, step_size=0.1, tol=1e-10, max_steps=100000):
    path = [start_coords]
    coords = np.array(start_coords)

    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
    # for _ in range(max_steps):
        _,grad = PES(coords)
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
        energy, grad = PES(coords)
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

def steepest_descent_momentum(start_coords, step_size=0.3, tol=1e-10, max_steps=10000, momentum=0.75):
    path = [start_coords]
    coords = np.array(start_coords)
    prev_coords = np.array(start_coords)  # La posición anterior (inicialmente igual a start_coords)
    prev_grad = np.zeros_like(coords)  # Gradiente del paso anterior

    # for _ in range(max_steps):
    for _ in tqdm(range(max_steps), desc="Optimizing", ncols=100, unit="step"):
        energy, grad = PES(coords)  # Obtener la energía y el gradiente
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


# Límites de disociación
start1 = np.array([15.,15.,1.40])  # Sustituye con valores reales
start2 = np.array([2.7,15.,15.5])

# Generar caminos desde ambos extremos
# path1 = steepest_descent(start1)
# path2 = steepest_descent(start2)
path1 = steepest_descent_momentum(start1)
path2 = steepest_descent_momentum(start2)
start1_from1 = path1[-1]
start2_from2 = path2[-1]

path1_to2 = steepest_crescent_smooth(start1_from1)
path2_to1 = steepest_crescent_smooth(start2_from2)

# Invertir uno de los caminos para facilitar la unión
path2 = path2[::-1]
path2_to1 = path2_to1[::-1]

# Unir caminos cuando la distancia entre sus últimos puntos sea pequeña
while np.linalg.norm(path1_to2[-1] - path2_to1[0]) < 1e-3:
    mid_point = (path1_to2[-1] + path2_to1[0]) / 2
    path1_to2 = np.vstack([path1_to2, mid_point])
    path2_to1 = np.vstack([mid_point, path2_to1])

final_path = np.vstack([path1, path1_to2, path2_to1, path2])
final_path = np.vstack([path1_to2, path2_to1])

# Crear un array para almacenar las coordenadas y la energía
path_with_energy = []

for coords in final_path:
    energy, _ = PES(coords)  # Obtener la energía para cada geometría
    path_with_energy.append(np.append(coords, energy))  # Añadir la energía a las coordenadas

# Convertir a array NumPy para guardar en un archivo
path_with_energy = np.array(path_with_energy)

# Guardar con encabezado para mejor comprensión
np.savetxt("minimum_energy_path.dat", path_with_energy)

# Extraer coordenadas y energías
r1, r2, r3, energy = path_with_energy.T

# Graficar la evolución de la energía a lo largo del camino
plt.figure(figsize=(8, 5))
plt.plot(r1-r2,energy, marker="o", linestyle="-", color="black", markersize=3, label="r1-r2")
plt.plot(r2-r1,energy, marker="o", linestyle="-", color="r", markersize=3, label="r2-r1")
plt.plot(r1-r3,energy, marker="o", linestyle="-", color="g", markersize=3, label="r1-r3")
plt.plot(r2-r3,energy, marker="o", linestyle="-", color="b", markersize=3, label="r2-r3")
plt.plot(r3-r1,energy, marker="o", linestyle="-", color="purple", markersize=3, label="r3-r1")
plt.plot(r3-r2,energy, marker="o", linestyle="-", color="orange", markersize=3, label="r3-r2")
# plt.plot(r1,energy, marker="o", linestyle="-", color="b", markersize=3, label="r1")
# plt.plot(r2,energy, marker="o", linestyle="-", color="r", markersize=3, label="r2")
# plt.plot(r3,energy, marker="o", linestyle="-", color="g", markersize=3, label="r3")

plt.xlabel("Paso en el camino")
plt.ylabel("Energía (u.a.)")
plt.title("Evolución de la Energía en el Camino de Mínima Energía")
plt.legend()
plt.grid()
# Anotar la diferencia de energía entre el primer y último valores en eV
delta_E = (energy[-1] - energy[0]) * 27.2114
plt.annotate(f"$\Delta E$ = {delta_E:.2f} eV", xy=(0.5, 0.95), xycoords="axes fraction",
             ha="center", va="top", size=12)


plt.show()