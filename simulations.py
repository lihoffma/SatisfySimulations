import pygame
import math
import random
import imageio
import colorsys
import numpy as np
import pygame
import math
import random
import imageio
import colorsys

def bouncing_ball_simulation(
    width,
    height,
    circle_initial_radius,
    ball_initial_radius,
    fps,
    gravity,
    shrink_per_collision,
    reset_pause_seconds,
    reset_growth_seconds,
    output_file
):
    """Simulation d'une balle rebondissant dans un cercle jusqu'à ce que la bille atteigne la taille du cercle."""

    # --- Constantes ---
    black, white = (0, 0, 0), (255, 255, 255)
    circle_center = (width // 2, height // 2)

    # --- Init pygame ---
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    # --- États initiaux ---
    circle_radius = circle_initial_radius
    ball_radius = ball_initial_radius
    ball_x, ball_y = circle_center[0] + 100, circle_center[1] - 50

    speed = 5
    angle = random.uniform(0, 2 * math.pi)
    vx, vy = speed * math.cos(angle), speed * math.sin(angle)

    resetting = False
    reset_frames = 0
    reset_pause = reset_pause_seconds * fps
    reset_growth = reset_growth_seconds * fps
    saved_vx, saved_vy = vx, vy  # vitesse sauvegardée avant reset

    # --- Préparer l'écriture directe ---
    writer = imageio.get_writer(output_file, fps=fps)

    # --- Boucle principale ---
    running = True
    frame = 0
    while running:
        if not resetting:
            # Gravité
            vy += gravity

            # Déplacement
            ball_x += vx
            ball_y += vy

            # Distance au centre
            dx, dy = ball_x - circle_center[0], ball_y - circle_center[1]
            dist = math.hypot(dx, dy)

            # Collision avec le cercle
            if dist + ball_radius >= circle_radius:
                normal_angle = math.atan2(dy, dx)
                nx, ny = math.cos(normal_angle), math.sin(normal_angle)

                # Réflexion
                dot = vx * nx + vy * ny
                vx -= 2 * dot * nx
                vy -= 2 * dot * ny

                # Corriger la position (éviter chevauchement)
                overlap = (dist + ball_radius) - circle_radius
                ball_x -= overlap * nx
                ball_y -= overlap * ny

                # Rétrécissement du cercle
                circle_radius -= shrink_per_collision
                if circle_radius <= ball_radius:
                    saved_vx, saved_vy = vx, vy
                    ball_radius += 10
                    resetting, reset_frames = True, 0
                    vx, vy = 0, 0  # arrêt pendant reset
        else:
            reset_frames += 1
            if reset_frames >= reset_pause:
                # Croissance progressive du cercle
                t = min((reset_frames - reset_pause) / reset_growth, 1.0)
                circle_radius = int(ball_radius + t * (circle_initial_radius - ball_radius))
                if t >= 1.0:
                    resetting = False
                    ball_x, ball_y = circle_center
                    vx, vy = saved_vx, saved_vy

        # --- Couleur arc-en-ciel ---
        hue = (frame % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        ball_color = (int(r * 255), int(g * 255), int(b * 255))

        # --- Dessin ---
        screen.fill(black)
        pygame.draw.circle(screen, white, circle_center, circle_radius, 2)
        pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
        pygame.display.flip()

        # Sauvegarde de frame directement dans le fichier vidéo
        frame_data = pygame.surfarray.array3d(screen).swapaxes(0, 1)
        writer.append_data(frame_data)

        clock.tick(fps)
        frame += 1

        # --- Condition d'arrêt ---
        if ball_radius >= circle_initial_radius:
            running = False

    pygame.quit()
    writer.close()  # fermer le fichier vidéo proprement
    print(f"Done: {output_file}")

def bouncing_ball_simulation_colored(
    width,
    height,
    circle_initial_radius,
    ball_initial_radius,
    fps,
    gravity,
    shrink_per_collision,
    reset_pause_seconds,
    reset_growth_seconds,
    view_pattern_seconds,
    output_file
):
    """Simulation d'une balle rebondissant dans un cercle avec effet de remplissage coloré."""

    # --- Constantes ---
    black, white = (0, 0, 0), (255, 255, 255)
    circle_center = (width // 2, height // 2)

    # --- Init pygame ---
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    # Surface pour accumuler la trace
    trail_surface = pygame.Surface((width, height))
    trail_surface.fill(black)
    
    # --- États initiaux ---
    circle_radius = circle_initial_radius
    ball_radius = ball_initial_radius
    ball_x, ball_y = circle_center[0], circle_center[1] - circle_radius + ball_radius*2

    speed = 7
    angle = random.uniform(-math.pi/4, -3*math.pi/4)  # Départ vers le bas
    vx, vy = speed * math.cos(angle), speed * math.sin(angle)

    resetting = False
    reset_frames = 0
    reset_pause = reset_pause_seconds * fps
    reset_growth = reset_growth_seconds * fps
    view_pattern = view_pattern_seconds * fps
    saved_vx, saved_vy = vx, vy

    viewing_pattern = False
    view_frames = 0

    writer = imageio.get_writer(output_file, fps=fps)
    frame = 0

    while True:
        if viewing_pattern:
            view_frames += 1
            if view_frames >= view_pattern:
                viewing_pattern = False
                view_frames = 0
                trail_surface.fill(black)
                ball_x, ball_y = circle_center
                vx, vy = saved_vx, saved_vy
        elif not resetting:
            # Gravité et déplacement
            vy += gravity
            ball_x += vx
            ball_y += vy

            # Distance au centre
            dx, dy = ball_x - circle_center[0], ball_y - circle_center[1]
            dist = math.hypot(dx, dy)

            # Couleur arc-en-ciel
            hue = (frame % 360) / 360.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
            ball_color = (int(r * 255), int(g * 255), int(b * 255))

            # Dessiner la trace
            pygame.draw.circle(trail_surface, ball_color, (int(ball_x), int(ball_y)), ball_radius)

            # Collision avec le cercle
            if dist + ball_radius >= circle_radius:
                normal_angle = math.atan2(dy, dx)
                nx, ny = math.cos(normal_angle), math.sin(normal_angle)
                dot = vx * nx + vy * ny
                vx -= 2 * dot * nx
                vy -= 2 * dot * ny

                # Correction position
                overlap = (dist + ball_radius) - circle_radius
                ball_x -= overlap * nx
                ball_y -= overlap * ny

                circle_radius -= shrink_per_collision
                if circle_radius <= ball_radius:
                    saved_vx, saved_vy = vx, vy
                    ball_radius += 10
                    resetting = True
                    reset_frames = 0
                    vx, vy = 0, 0
        else:
            reset_frames += 1
            if reset_frames >= reset_pause:
                t = min((reset_frames - reset_pause) / reset_growth, 1.0)
                circle_radius = int(ball_radius + t * (circle_initial_radius - ball_radius))
                if t >= 1.0:
                    resetting = False
                    viewing_pattern = True
                    view_frames = 0

        # --- Dessin ---
        screen.fill(black)
        
        # Créer et appliquer le masque circulaire
        mask = pygame.Surface((width, height))
        mask.fill(black)
        pygame.draw.circle(mask, white, circle_center, circle_radius)
        
        # Copier la trace masquée sur l'écran
        screen_copy = trail_surface.copy()
        screen_copy.blit(mask, (0, 0), special_flags=pygame.BLEND_MULT)
        screen.blit(screen_copy, (0, 0))
        
        # Dessiner le cercle et la balle
        pygame.draw.circle(screen, white, circle_center, circle_radius, 2)
        pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
        
        pygame.display.flip()
        
        frame_data = pygame.surfarray.array3d(screen).swapaxes(0, 1)
        writer.append_data(frame_data)
        
        clock.tick(fps)
        frame += 1

        if ball_radius >= circle_initial_radius:
            break

    pygame.quit()
    writer.close()
    print(f"Done: {output_file}")





def bouncing_ball_simulation_gradient(
    width,
    height,
    circle_radius,
    ball_initial_radius,
    fps,
    gravity,
    chaos_interval,
    output_file,
    SHRINK_PER_COLLISION,
    ball_radius_var
):
    """
    Simulation d'une balle rebondissant dans un cercle qui révèle un gradient.
    La balle efface le masque noir pour révéler le gradient en dessous.
    """
    circle_initial_radius = circle_radius
    # --- Constantes ---
    black, white = (0, 0, 0), (255, 255, 255)
    circle_center = (width // 2, height // 2)
    MAX_SPEED = 12
    MIN_SPEED = 5
    
    # --- Init pygame et police ---
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font_size = min(width, height) // 20
    font = pygame.font.Font(None, font_size)
    
    def create_gradient():
        """Crée un nouveau gradient aléatoire"""
        color1 = [random.randint(0, 255) for _ in range(3)]
        color2 = [random.randint(0, 255) for _ in range(3)]
        gradient_surface = pygame.Surface((width, height))
        
        for y in range(height):
            ratio = y / height
            color = [int(c1 + (c2 - c1) * ratio) 
                    for c1, c2 in zip(color1, color2)]
            pygame.draw.line(gradient_surface, color, (0, y), (width, y))
        
        return gradient_surface
    
    def calculate_revealed_percentage(mask_surface, circle_center, circle_radius):
        """Calcule le pourcentage de zone révélée dans le cercle"""
        # Créer un masque pour le cercle complet
        circle_mask = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(circle_mask, (255, 255, 255, 255), circle_center, circle_radius)
        
        # Convertir en tableau numpy pour le calcul
        circle_array = pygame.surfarray.pixels_alpha(circle_mask)
        mask_array = pygame.surfarray.pixels_alpha(mask_surface)
        
        # Calculer les pixels dans le cercle et les pixels révélés
        total_pixels = np.sum(circle_array > 0)
        revealed_pixels = np.sum((circle_array > 0) & (mask_array < 255))
        
        # Libérer les surfaces
        del circle_array
        del mask_array
        
        # Calculer le pourcentage
        if total_pixels == 0:
            return 0
        return (revealed_pixels / total_pixels) * 100

    def add_chaos(vx, vy):
        """Ajoute du chaos au mouvement de la balle"""
        current_speed = math.hypot(vx, vy)
        chaos_types = [
            ("gravity_flip", lambda: (vx, -vy * 0.9)),
            ("rotation", lambda: (
                vx * math.cos(random.uniform(-math.pi/6, math.pi/6)) - 
                vy * math.sin(random.uniform(-math.pi/6, math.pi/6)),
                vx * math.sin(random.uniform(-math.pi/6, math.pi/6)) + 
                vy * math.cos(random.uniform(-math.pi/6, math.pi/6))
            )),
            ("speed_adjust", lambda: (
                vx * random.uniform(0.95, 1.05),
                vy * random.uniform(0.95, 1.05)
            ) if MIN_SPEED <= current_speed <= MAX_SPEED else (vx, vy))
        ]
        
        chosen_type = random.choice(chaos_types)
        return chosen_type[1]()
    
    # --- États initiaux ---
    ball_radius = ball_initial_radius
    ball_x, ball_y = circle_center[0], circle_center[1] - circle_radius + ball_radius*2
    speed = 7
    angle = random.uniform(-math.pi/4, -3*math.pi/4)
    vx, vy = speed * math.cos(angle), speed * math.sin(angle)
    
    # Surfaces
    gradient = create_gradient()
    mask_surface = pygame.Surface((width, height), pygame.SRCALPHA)
    mask_surface.fill((0, 0, 0, 255))
    
    # Ajouter après les compteurs existants
    frame = 0
    chaos_frames = 0
    chaos_trigger = chaos_interval * fps
    time_since_last_percent = 0  # Compteur pour le temps écoulé
    last_percentage = 0  # Dernier pourcentage enregistré
    timeout_frames = 5 * fps  # 2 secondes à 60 fps
    
    writer = imageio.get_writer(output_file, fps=fps)
    
    while ball_radius <= circle_initial_radius/2:
        # Ajout de chaos périodique
        chaos_frames += 1
        if chaos_frames >= chaos_trigger:
            vx, vy = add_chaos(vx, vy)
            chaos_frames = 0
        
        # Physique
        vy += gravity
        ball_x += vx
        ball_y += vy
        
        # Distance au centre
        dx, dy = ball_x - circle_center[0], ball_y - circle_center[1]
        dist = math.hypot(dx, dy)
        
        # Collision avec le cercle
        if dist + ball_radius >= circle_radius:
            normal_angle = math.atan2(dy, dx)
            nx, ny = math.cos(normal_angle), math.sin(normal_angle)
            dot = vx * nx + vy * ny
            vx -= 2 * dot * nx
            vy -= 2 * dot * ny
            
            # Léger boost au rebond
            current_speed = math.hypot(vx, vy)
            if current_speed < MAX_SPEED:
                boost = random.uniform(1.01, 1.03)
                vx *= boost
                vy *= boost
            
            # Correction position
            overlap = (dist + ball_radius) - circle_radius
            ball_x -= overlap * nx
            ball_y -= overlap * ny
            
            # Rétrécir le cercle à chaque collision
            circle_radius = circle_radius - SHRINK_PER_COLLISION
        
        
        # Effacer le masque au passage de la balle
        pygame.draw.circle(mask_surface, (0, 0, 0, 0), 
                         (int(ball_x), int(ball_y)), ball_radius)
        
        # Calcul du pourcentage révélé
        percentage = calculate_revealed_percentage(mask_surface, circle_center, circle_radius)
        
        # Vérifier si le pourcentage a changé
        if abs(percentage - last_percentage) < 0.09:  # Si pas de changement significatif
            time_since_last_percent += 1
            if time_since_last_percent >= timeout_frames:
                percentage = 100  # Force le pourcentage à 100%
        else:
            time_since_last_percent = 0  # Réinitialiser le compteur
            last_percentage = percentage

        # Vérifier si on doit passer au niveau suivant
        if percentage >= 99.99:
            time_since_last_percent = 0  # Réinitialiser le compteur
            ball_radius += ball_radius_var
            circle_radius = 300
            gradient = create_gradient()
            mask_surface.fill((0, 0, 0, 255))
            ball_x, ball_y = circle_center[0], circle_center[1] - circle_radius + ball_radius*2
            angle = random.uniform(-math.pi/4, -3*math.pi/4)
            vx, vy = speed * math.cos(angle), speed * math.sin(angle)
            last_percentage = 0  # Réinitialiser le dernier pourcentage

        # Couleur pour la balle
        ball_color = (0, 0, 0)
        pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
        
        # Dessin
        screen.fill(black)
        screen.blit(gradient, (0, 0))
        screen.blit(mask_surface, (0, 0))
        
        # Créer un masque pour l'extérieur du cercle
        outer_mask = pygame.Surface((width, height), pygame.SRCALPHA)
        outer_mask.fill((0, 0, 0, 255))  # Remplir en noir opaque
        # Créer un trou transparent dans le masque
        pygame.draw.circle(outer_mask, (0, 0, 0, 0), circle_center, circle_radius)
        screen.blit(outer_mask, (0, 0))  # Appliquer le masque extérieur
        
        # Dessiner le cercle blanc et la balle
        pygame.draw.circle(screen, white, circle_center, circle_radius, 2)
        pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
        
        # Affichage du pourcentage
        text = font.render(f"{percentage:.1f}%", True, white)
        text_rect = text.get_rect()
        text_rect.bottomright = (width - 20, height - 20)
        
        # Fond semi-transparent pour le texte
        padding = 10
        bg_rect = text_rect.inflate(padding * 2, padding * 2)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.fill(black)
        bg_surface.set_alpha(128)
        screen.blit(bg_surface, bg_rect)
        screen.blit(text, text_rect)
        
        pygame.display.flip()
        frame_data = pygame.surfarray.array3d(screen).swapaxes(0, 1)
        writer.append_data(frame_data)
        
        clock.tick(fps)
        frame += 1
    
    pygame.quit()
    writer.close()
    print(f"Done: {output_file}")


# Exemple d’exécution
def main():

    format = "v"  # "h" pour horizontal, "v" pour vertical

    for format in ["h","v"]:

        if format == "h":
            width, height = 1080, 720
            output_dir = "Simulations/Rebond/Horizontal"
        else:
            width, height = 720, 1080
            output_dir = "Simulations/Rebond/Vertical"
        
        
        for i in range(1):

            output_file = f"{output_dir}/simulation{i}.mp4"

            # circ_rad = 400
            # ball_rad = 3
            # grav = round(random.uniform(0.1, 0.3), 2)
            # shrink = 2

            # bouncing_ball_simulation(width=1080,
            # height=720,
            # circle_initial_radius=circ_rad,
            # ball_initial_radius=ball_rad,
            # fps=60,
            # gravity=grav,
            # shrink_per_collision=shrink,
            # reset_pause_seconds=1,
            # reset_growth_seconds=2,
            # output_file=f"simulation{i}.mp4")

            # bouncing_ball_simulation_colored(
            #     width,
            #     height,
            #     circle_initial_radius=circ_rad,
            #     ball_initial_radius=ball_rad,
            #     fps=60,
            #     gravity=grav,
            #     shrink_per_collision=shrink,
            #     reset_pause_seconds=1,
            #     reset_growth_seconds=2,
            #     view_pattern_seconds=3,  # Pause de 3 secondes pour observer le motif
            #     output_file=output_file
            # )

            # chaos_interval = 20  # entre 0.5 et 2 secondes
            # ball_radius_var = 5  # Variation aléatoire du rayon de la balle entre 1 et 5
            # bouncing_ball_simulation_gradient(  width,
            #                                     height,
            #                                     circ_rad,
            #                                     ball_rad,
            #                                     fps= 60,
            #                                     gravity=grav,
            #                                     chaos_interval = chaos_interval,
            #                                     output_file = output_file,
            #                                     SHRINK_PER_COLLISION = shrink,
            #                                     ball_radius_var=ball_radius_var
            #                                     )            


# main()

def particules_simulation(format, duree, num_particles, num_attractors, speed, trail_alpha, output_file):

    if format == "h":
            WIDTH, HEIGHT = 1080, 720
    else:
            WIDTH, HEIGHT = 720, 1080


    # --- Paramètres ---
    NUM_PARTICLES = num_particles
    NUM_ATTRACTORS = num_attractors
    SPEED = speed  # vitesse des particules
    TRAIL_ALPHA = trail_alpha  # 0 = pas de traînées, 255 = pas de transparence
    FPS = 60
    TOTAL_FRAMES = duree * FPS

    # --- Initialisation ---
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulation Satisfying - Particules")
    clock = pygame.time.Clock()
    writer = imageio.get_writer(output_file, fps=FPS)

    # Surface pour les traînées
    trail_surface = pygame.Surface((WIDTH, HEIGHT))
    trail_surface.set_alpha(TRAIL_ALPHA)
    trail_surface.fill((0, 0, 0))

    # --- Classes ---
    class Particle:
        def __init__(self):
            self.x = random.uniform(0, WIDTH)
            self.y = random.uniform(0, HEIGHT)
            self.vx = 0
            self.vy = 0

        def update(self, attractors, t):
            ax, ay = 0, 0
            for (ax_pos, ay_pos) in attractors:
                dx = ax_pos - self.x
                dy = ay_pos - self.y
                dist2 = dx * dx + dy * dy + 0.1
                force = 100 / dist2  # intensité de l’attraction
                ax += dx * force
                ay += dy * force

            # mise à jour vitesse + position
            self.vx = (self.vx + ax * SPEED) * 0.95
            self.vy = (self.vy + ay * SPEED) * 0.95
            self.x += self.vx
            self.y += self.vy

            # rebond sur les bords
            if self.x < 0 or self.x > WIDTH:
                self.vx *= -1
            if self.y < 0 or self.y > HEIGHT:
                self.vy *= -1

            # couleur dynamique en fonction du temps et vitesse
            speed = math.sqrt(self.vx**2 + self.vy**2)
            r = int((math.sin(t*0.01 + speed) + 1) * 127)
            g = int((math.sin(t*0.015 + self.x*0.01) + 1) * 127)
            b = int((math.sin(t*0.02 + self.y*0.01) + 1) * 127)
            color = (r, g, b)

            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 2)


    # --- Création des particules ---
    particles = [Particle() for _ in range(NUM_PARTICLES)]

    # --- Boucle principale ---
    t = 0
    frame_count = 0
    
    while frame_count < TOTAL_FRAMES:
        clock.tick(FPS)
        t += 1

        # effet traînée
        screen.blit(trail_surface, (0, 0))

        # attracteurs qui bougent
        attractors = []
        for i in range(NUM_ATTRACTORS):
            ax = WIDTH // 2 + math.sin(t*0.01 + i*2) * 200
            ay = HEIGHT // 2 + math.cos(t*0.008 + i*3) * 200
            attractors.append((ax, ay))

        # mise à jour & dessin particules
        for p in particles:
            p.update(attractors, t)

        pygame.display.flip()
        
        # Capture de la frame pour la vidéo
        frame_data = pygame.surfarray.array3d(screen).swapaxes(0, 1)
        writer.append_data(frame_data)
        
        frame_count += 1

    pygame.quit()
    writer.close()
    print(f"✅ Vidéo sauvegardée : {output_file}")


def main2():
    for format in ["h","v"]:
        if format == "h":
            width, height = 1080, 720
            output_dir = "Simulations/BlackHole/Horizontal"
        else:
            width, height = 720, 1080
            output_dir = "Simulations/BlackHole/Vertical"
        for i, n_duree in zip(range(5), [180, 250, 360, 250, 720]):
            output_file = f"{output_dir}/simulation{i}.mp4"
            n_particles = random.randint(100, 700)
            n_attractors = random.randint(2, 5)
            n_speed = round(random.uniform(0.3, 0.7), 2)
            n_trail_alpha = random.randint(15, 40)
            particules_simulation(format, duree=n_duree, num_particles=n_particles, num_attractors=n_attractors, speed=n_speed, trail_alpha=n_trail_alpha, output_file=output_file)

# main2()

def test():
    format = "v"  # "h" pour horizontal, "v" pour vertical
    if format == "h":
            width, height = 1080, 720
            output_dir = "Simulations/BlackHole/Horizontal"
    else:
            width, height = 720, 1080
            output_dir = "Simulations/BlackHole/Vertical"
    n_particles = random.randint(100, 700)
    n_attractors = random.randint(2, 5)
    n_speed = round(random.uniform(0.3, 0.7), 2)
    n_trail_alpha = random.randint(15, 40)
    particules_simulation(format, duree=720, num_particles=n_particles, num_attractors=n_attractors, speed=n_speed, trail_alpha=n_trail_alpha, output_dir=output_dir)

# test()


def test2():
    import pygame
    import random
    import math

    # Initialisation
    pygame.init()
    WIDTH, HEIGHT = 1200, 800
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Système solaire avec comètes")

    # Couleurs
    BLACK = (0, 0, 0)

    # Constante gravitationnelle simplifiée
    G = 0.4  

    # Nombre de comètes
    N_COMETS = 3  

    # Classe Planète (orbite elliptique autour du Soleil)
    class Planet:
        def __init__(self, a, b, radius, speed, angle=0, mass=None):
            """
            a, b : demi-grand axe et demi-petit axe (ellipse)
            radius : taille de la planète
            speed : vitesse angulaire
            angle : position initiale
            """
            self.a = a
            self.b = b
            self.radius = radius
            self.speed = speed
            self.angle = angle
            self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            self.mass = mass if mass else radius * 80
            self.x = WIDTH // 2 + int(self.a * math.cos(self.angle))
            self.y = HEIGHT // 2 + int(self.b * math.sin(self.angle))

        def update(self):
            self.angle += self.speed
            self.x = WIDTH // 2 + int(self.a * math.cos(self.angle))
            self.y = HEIGHT // 2 + int(self.b * math.sin(self.angle))

        def draw(self, win):
            pygame.draw.circle(win, self.color, (self.x, self.y), self.radius)

    # Classe Comète
    class Comet:
        def __init__(self, x, y, vx, vy, color=(255, 255, 255)):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.color = color
            self.trail = []
        
        def update(self, planets):
            ax, ay = 0, 0
            for planet in planets:
                dx = planet.x - self.x
                dy = planet.y - self.y
                dist_sq = dx**2 + dy**2
                dist = math.sqrt(dist_sq)
                if dist > planet.radius:  
                    force = G * planet.mass / dist_sq
                    ax += force * dx / dist
                    ay += force * dy / dist

            # Mise à jour de la vitesse et position
            self.vx += ax
            self.vy += ay
            self.x += self.vx
            self.y += self.vy

            # Traînée lumineuse
            self.trail.append((int(self.x), int(self.y)))
            if len(self.trail) > 150:
                self.trail.pop(0)

        def draw(self, win):
            for i, pos in enumerate(self.trail):
                pygame.draw.circle(win, self.color, pos, 2)
            pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), 4)

    # Soleil au centre
    sun = Planet(0, 0, 60, 0)
    sun.color = (255, 255, 0)
    sun.mass = 20000  # Soleil massif

    # Création des planètes avec orbites elliptiques
    planets = [sun]
    for i in range(6):
        a = random.randint(120 + i*70, 150 + i*80)  # demi-grand axe
        b = random.randint(80 + i*50, 120 + i*60)   # demi-petit axe
        radius = random.randint(15, 35)
        speed = random.uniform(0.005, 0.02) * (1 if i % 2 == 0 else -1)  # vitesses différentes
        angle = random.uniform(0, 2*math.pi)
        planets.append(Planet(a, b, radius, speed, angle))

    # Création des comètes
    comets = []
    for _ in range(N_COMETS):
        x = random.randint(50, WIDTH-50)
        y = random.randint(50, HEIGHT-50)
        vx = random.uniform(-3, 3)
        vy = random.uniform(-3, 3)
        comets.append(Comet(x, y, vx, vy, color=(255, 255, 255)))

    # Boucle principale
    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(60)
        win.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Mise à jour des planètes
        for planet in planets[1:]:  
            planet.update()

        # Mise à jour des comètes
        for comet in comets:
            comet.update(planets)

        # Dessin Soleil et planètes
        for planet in planets:
            planet.draw(win)

        # Dessin comètes
        for comet in comets:
            comet.draw(win)

        pygame.display.update()

    pygame.quit()

test2()

