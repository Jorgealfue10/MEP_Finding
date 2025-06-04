import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

data = np.loadtxt("minimum_energy_path.dat")
if data.size == 0:
    raise ValueError("The file 'minimum_energy_path.dat' is empty or does not exist.")

r12 = data[:, 0]
r13 = data[:, 1]
r23 = data[:, 2]
e = (data[:, 3]-data[0, 3])*27.2114  # Subtract the first energy value to set the minimum at zero

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(r23-r12, e, color='blue', linewidth=2, label=r"MEP 1 $^3$A''")
ax.hlines(0, -10, 10, color='red', linestyle='--', linewidth=2,zorder=-10)

# x1 = 9.5
# dx = 0.0
# y1 = 0.0
# dy = 1.15

# ax.arrow(x1, y1, dx, dy, head_width=0.2, head_length=0.05, fc='black', ec='black')

ax.annotate(
    '1.20 eV',
    xy=(9.85,0.0),      # extremo derecho
    xytext=(9.25, 1.25),  # extremo izquierdo
    arrowprops=dict(arrowstyle='<->', lw=2, color='black')
)

ax.set_xlabel(r"$r_{23} - r_{12}$ (Bohr)", fontsize=14)
ax.set_ylabel("Energy (eV)", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
# ax.grid(True)
ax.legend(fontsize=12)
plt.tight_layout()
output_file = "minimum_energy_path_plot.png"
if os.path.exists(output_file):
    os.remove(output_file)
plt.savefig(output_file, dpi=300, transparent=True)
plt.show()