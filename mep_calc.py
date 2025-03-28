import numpy as np
from scipy.optimize import minimize
from diatPHM import *
from diatH2 import *
from body3 import *

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

# Gradiente de la PES (derivadas parciales)
# def gradient_PES(coords, delta=1e-4):
#     grad = np.zeros(3)
#     for i in range(3):
#         step = np.zeros(3)
#         step[i] = delta
#         grad[i] = (PES(coords + step) - PES(coords - step)) / (2 * delta)
#     return grad

# Método de Steepest Descent
def steepest_descent(start_coords, step_size=0.01, tol=1e-5, max_steps=1000):
    path = [start_coords]
    coords = np.array(start_coords)

    for _ in range(max_steps):
        _,grad = PES(coords)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break  # Nos detenemos si el gradiente es muy pequeño
        
        # Paso en dirección de mayor descenso
        coords = coords - step_size * grad / grad_norm
        path.append(coords)

    return np.array(path)

# Límites de disociación
start1 = np.array([18.,18.,1.40])  # Sustituye con valores reales
start2 = np.array([2.70,18.,17.5])

# Generar caminos desde ambos extremos
path1 = steepest_descent(start1)
path2 = steepest_descent(start2)

# Invertir uno de los caminos para facilitar la unión
path2 = path2[::-1]

# Unir caminos cuando la distancia entre sus últimos puntos sea pequeña
while np.linalg.norm(path1[-1] - path2[0]) > 1e-3:
    mid_point = (path1[-1] + path2[0]) / 2
    path1 = np.vstack([path1, mid_point])
    path2 = np.vstack([mid_point, path2])

final_path = np.vstack([path1, path2])

# Guardar el camino de mínima energía
np.savetxt("minimum_energy_path.dat", final_path)
