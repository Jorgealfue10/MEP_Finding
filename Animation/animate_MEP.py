from manim import *
import numpy as np

class TestScene(ThreeDScene):
    def construct(self):
        self.elapsed_time = 0

        # Simulación de coordenadas de prueba
        r_p_h1 = np.linspace(1, 2, 100)  # Variando el enlace P-H1
        r_p_h2 = np.linspace(1, 2, 100)  # Variando el enlace P-H2
        r_h1_h2 = np.linspace(1.5, 2.5, 100)  # Variando H1-H2
        num_frames = len(r_p_h1)

        # Esfera central (átomo de fósforo)
        atom_p = Sphere(radius=0.2, color=ORANGE).move_to([0, 0, 0])

        # Hidrógenos (simples para prueba)
        atom_h1 = Sphere(radius=0.1, color=BLUE).move_to([r_p_h1[0], 0, 0])
        atom_h2 = Sphere(radius=0.1, color=BLUE).move_to([-r_p_h2[0], 0, 0])

        self.add(atom_p, atom_h1, atom_h2)

        def update_h1(mob, dt):
            self.elapsed_time += dt
            frame = min(int(self.elapsed_time * num_frames / 4), num_frames - 1)
            mob.move_to([r_p_h1[frame], 0, 0])

        def update_h2(mob, dt):
            frame = min(int(self.elapsed_time * num_frames / 4), num_frames - 1)
            mob.move_to([-r_p_h2[frame], 0, 0])

        atom_h1.add_updater(update_h1)
        atom_h2.add_updater(update_h2)

        self.wait(4)

class MEP2(ThreeDScene):
    def construct(self):
        self.elapsed_time = 0

        self.camera.background_color = None  # Fondo transparente
        # Leer los datos del archivo MEP
        r_p_h1, r_p_h2, r_h1_h2, energy = np.loadtxt("gs.dat", delimiter=' ', unpack=True)
        num_frames = len(r_p_h1)

        # Función para calcular coordenadas cartesianas en 3D
        def get_cartesian_coordinates(r_p_h1, r_p_h2, r_h1_h2):
            phosphorus_pos = np.array([0, 0, 0])

            # H1 en el eje X
            h1_pos = np.array([r_p_h1, 0, 0])

            # Cálculo de H2 basado en distancias relativas
            x2 = (r_p_h2**2 - r_h1_h2**2 + r_p_h1**2) / (2 * r_p_h1)
            y2_squared = r_p_h2**2 - x2**2
            y2 = np.sqrt(y2_squared) if y2_squared >= 0 else 0  # Evitar valores imaginarios
            z2 = np.sqrt(max(0, r_p_h2**2 - x2**2 - y2**2))  # Evitar NaN

            h2_pos = np.array([x2, y2, z2])

            return phosphorus_pos, h1_pos, h2_pos

        # Posiciones iniciales
        p_pos, h1_pos, h2_pos = get_cartesian_coordinates(r_p_h1[0], r_p_h2[0], r_h1_h2[0])

        # Crear esferas para los átomos
        atom_p = Sphere(radius=0.2, color=ORANGE).move_to(p_pos)
        atom_h1 = Sphere(radius=0.1, color=BLUE).move_to(h1_pos)
        atom_h2 = Sphere(radius=0.1, color=BLUE).move_to(h2_pos)

        # Calcular el centro de masas
        def calculate_center_of_mass(p_pos, h1_pos, h2_pos):
            # Para simplificar, vamos a usar las posiciones y asumimos que las masas son iguales (tamaño de las esferas)
            return (p_pos + h1_pos + h2_pos) / 3

        # Obtener el centro de masas inicial
        center_of_mass = calculate_center_of_mass(p_pos, h1_pos, h2_pos)
        # Vectores de los átomos para calcular el plano
        vector_p_h1 = h1_pos - p_pos
        vector_p_h2 = h2_pos - p_pos
        # Producto cruzado para obtener el vector normal al plano
        normal_vector = np.cross(vector_p_h1, vector_p_h2)
        # Actualizar la cámara para que siga el centro de masas
        self.move_camera(
            frame_center=p_pos,
            orientation=normal_vector,
            distance=17,
        )

        self.add(atom_p, atom_h1, atom_h2)

        self.set_camera_orientation(phi=45 * DEGREES, theta=45 * DEGREES)
        # self.move_camera(frame_center=center_of_mass)

        # Función updater para mover los átomos
        def update_atoms(mob, dt):
            self.elapsed_time += dt
            # frame = min(int(self.elapsed_time * num_frames / 10), num_frames - 1)
            frame = 242000+int(self.elapsed_time*300)
            # print(frame,r_p_h1[frame], r_p_h2[frame], r_h1_h2[frame])
            _, new_h1_pos, new_h2_pos = get_cartesian_coordinates(r_p_h1[frame], r_p_h2[frame], r_h1_h2[frame])

            # # print(r_p_h1[frame], r_p_h2[frame], r_h1_h2[frame])
            atom_p.set_color(ORANGE)
            atom_h1.set_color(BLUE)
            atom_h2.set_color(BLUE)
            atom_h1.move_to(new_h1_pos)
            atom_h2.move_to(new_h2_pos)

            # Actualizar el centro de masas
            center_of_mass = calculate_center_of_mass(p_pos, new_h1_pos, new_h2_pos)

             # Vectores de los átomos para calcular el plano
            vector_p_h1 = new_h1_pos - p_pos
            vector_p_h2 = new_h2_pos - p_pos

            # Producto cruzado para obtener el vector normal al plano
            normal_vector = np.cross(vector_p_h1, vector_p_h2)

            # # Actualizar la cámara para que siga el centro de masas
            # self.move_camera(
            #     frame_center=center_of_mass,
            #     # orientation=normal_vector,
            #     distance=15,
            # )

        # Agregar updaters solo a H1 y H2
        atom_h1.add_updater(update_atoms)
        atom_h2.add_updater(update_atoms)

        self.wait(10)

        atom_h1.clear_updaters()
        atom_h2.clear_updaters()
        self.wait(1)

        # # Obtener el centro de masas inicial
        # center_of_mass = calculate_center_of_mass(p_pos, h1_pos, h2_pos)

        # # Vectores de los átomos para calcular el plano
        # vector_p_h1 = h1_pos - p_pos
        # vector_p_h2 = h2_pos - p_pos

        # # Producto cruzado para obtener el vector normal al plano
        # normal_vector = np.cross(vector_p_h1, vector_p_h2)

        # self.move_camera(
        #     # frame_center=atom_p.get_center(),
        #     frame_center=center_of_mass,
        #     orientation=normal_vector,
        #     distance=15,
        #     # added_ops=[],
        # )

        # # Esperar para que la animación se vea
        # self.wait(10)


class MEPAnimation(ThreeDScene):
    def construct(self):
        self.elapsed_time = 0

        # Leer los datos del archivo MEP
        r_p_h1, r_p_h2, r_h1_h2, energy = np.loadtxt("minimum_energy_path.dat", delimiter=' ', unpack=True)
        
        # Función para calcular las posiciones cartesianas de H1 y H2 en 3D
        def get_cartesian_coordinates(r_p_h1, r_p_h2, r_h1_h2, angle_offset=0):
            # Posición de fósforo (P) es el origen
            phosphorus_pos = np.array([0, 0, 0])

            # H1 está a una distancia r_p_h1 en el eje X
            h1_pos = np.array([r_p_h1 * np.cos(angle_offset), r_p_h1 * np.sin(angle_offset), 0])

            # Calcular las coordenadas de H2 en 3D usando las distancias relativas
            x2 = (r_p_h2**2 - r_h1_h2**2 + r_p_h1**2) / (2 * r_p_h1)
            y2 = np.sqrt(r_p_h2**2 - x2**2)
            
            # Calcular la coordenada Z de H2
            z2 = np.sqrt(r_p_h2**2 - x2**2 - y2**2)

            # H2 está en las coordenadas (x2, y2, z2) en 3D
            h2_pos = np.array([x2, y2, z2])

            return phosphorus_pos, h1_pos, h2_pos

        p_pos,h1_pos,h2_pos = get_cartesian_coordinates(r_p_h1[0], r_p_h2[0], r_h1_h2[0])
        # Crear los átomos de fósforo y hidrógeno
        atom_p = Sphere(radius=0.2).move_to(p_pos)
        atom_h1 = Sphere(radius=0.1).move_to(h1_pos)
        atom_h2 = Sphere(radius=0.1).move_to(h2_pos)

        # Añadir los átomos a la escena
        self.add(atom_p, atom_h1, atom_h2)

                # Función de actualización para cada átomo
        def update_p(mob, dt):
            num_frames = len(r_p_h1)
            self.elapsed_time += dt
            frame = min(int(self.elapsed_time * num_frames / 4), num_frames - 1)
            phosphorus_pos, _, _ = get_cartesian_coordinates(r_p_h1[frame], r_p_h2[frame], r_h1_h2[frame])
            mob.move_to(phosphorus_pos)

        def update_h1(mob, dt):
            num_frames = len(r_p_h1)
            self.elapsed_time += dt
            frame = min(int(self.elapsed_time * num_frames / 4), num_frames - 1)
            _, h1_pos, _ = get_cartesian_coordinates(r_p_h1[frame], r_p_h2[frame], r_h1_h2[frame])
            mob.move_to(h1_pos)

        def update_h2(mob, dt):
            num_frames = len(r_p_h1)
            self.elapsed_time += dt
            frame = min(int(self.elapsed_time * num_frames / 4), num_frames - 1)
            _, _, h2_pos = get_cartesian_coordinates(r_p_h1[frame], r_p_h2[frame], r_h1_h2[frame])
            mob.move_to(h2_pos)

        # Añadir los updaters individuales
        atom_p.add_updater(update_p)
        atom_h1.add_updater(update_h1)
        atom_h2.add_updater(update_h2)

        # Iniciar la animación
        self.wait(4)

        # Detener los updaters después de la animación
        atom_p.clear_updaters()
        atom_h1.clear_updaters()
        atom_h2.clear_updaters()

        self.wait(1)