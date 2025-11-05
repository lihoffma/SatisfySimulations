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

# test2()


def test3():
    import pygame
    import random
    import numpy as np
    from PIL import Image

    # --- Paramètres ---
    TILE_SIZE = 3        # Taille d’un bloc (patch n×n pixels)
    OUTPUT_WIDTH = 30    # Largeur en nombre de blocs
    OUTPUT_HEIGHT = 30   # Hauteur en nombre de blocs
    IMAGE_PATH = "input.png"  # Image de base à fournir

    # Charger et convertir en tableau numpy
    img = Image.open(IMAGE_PATH).convert("RGB")
    pixels = np.array(img)
    h, w, _ = pixels.shape

    # --- Découper l’image en blocs de TILE_SIZE ---
    def extract_tiles():
        tiles = []
        for y in range(h - TILE_SIZE + 1):
            for x in range(w - TILE_SIZE + 1):
                patch = pixels[y:y+TILE_SIZE, x:x+TILE_SIZE]
                tiles.append(patch)
        return tiles

    tiles = extract_tiles()

    # Supprimer les doublons
    unique_tiles = []
    tile_ids = {}
    for t in tiles:
        key = t.tobytes()
        if key not in tile_ids:
            tile_ids[key] = len(unique_tiles)
            unique_tiles.append(t)

    # Fonction pour obtenir les bords d’une tuile
    def get_edges(tile):
        top = tile[0, :, :].tobytes()
        bottom = tile[-1, :, :].tobytes()
        left = tile[:, 0, :].tobytes()
        right = tile[:, -1, :].tobytes()
        return {"top": top, "bottom": bottom, "left": left, "right": right}

    edges = [get_edges(t) for t in unique_tiles]

    # Construire les règles de compatibilité
    compatibility = {i: {"top": [], "bottom": [], "left": [], "right": []} for i in range(len(unique_tiles))}
    for i, e1 in enumerate(edges):
        for j, e2 in enumerate(edges):
            if e1["top"] == e2["bottom"]:
                compatibility[i]["top"].append(j)
            if e1["bottom"] == e2["top"]:
                compatibility[i]["bottom"].append(j)
            if e1["left"] == e2["right"]:
                compatibility[i]["left"].append(j)
            if e1["right"] == e2["left"]:
                compatibility[i]["right"].append(j)

    # Convertir en surface pygame
    def tile_to_surface(tile):
        surf = pygame.Surface((TILE_SIZE, TILE_SIZE))
        pygame.surfarray.blit_array(surf, tile)
        return surf

    tile_surfaces = [tile_to_surface(t) for t in unique_tiles]

    # --- Grille WFC ---
    grid = [[list(range(len(unique_tiles))) for _ in range(OUTPUT_HEIGHT)] for _ in range(OUTPUT_WIDTH)]

    def is_collapsed(x, y):
        return len(grid[x][y]) == 1

    def collapse_cell(x, y):
        options = grid[x][y]
        if len(options) > 1:
            choice = random.choice(options)
            grid[x][y] = [choice]

    def propagate():
        changed = True
        while changed:
            changed = False
            for x in range(OUTPUT_WIDTH):
                for y in range(OUTPUT_HEIGHT):
                    if not grid[x][y]:
                        continue
                    options = grid[x][y]
                    for direction, (dx, dy) in {"top":(0,-1), "bottom":(0,1), "left":(-1,0), "right":(1,0)}.items():
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < OUTPUT_WIDTH and 0 <= ny < OUTPUT_HEIGHT:
                            neighbor_opts = grid[nx][ny]
                            if neighbor_opts:
                                allowed = set()
                                for opt in options:
                                    allowed.update(compatibility[opt][direction])
                                new_neighbor_opts = [o for o in neighbor_opts if o in allowed]
                                if set(new_neighbor_opts) != set(neighbor_opts):
                                    grid[nx][ny] = new_neighbor_opts
                                    changed = True

    # --- Pygame ---
    pygame.init()
    screen = pygame.display.set_mode((OUTPUT_WIDTH*TILE_SIZE, OUTPUT_HEIGHT*TILE_SIZE))
    pygame.display.set_caption("Wave Function Collapse - Contraintes")
    clock = pygame.time.Clock()

    running = True
    while running:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Choisir la cellule la moins contrainte (plus faible entropie)
        non_collapsed = [(x, y) for x in range(OUTPUT_WIDTH) for y in range(OUTPUT_HEIGHT) if not is_collapsed(x, y)]
        if non_collapsed:
            x, y = min(non_collapsed, key=lambda pos: len(grid[pos[0]][pos[1]]))
            collapse_cell(x, y)
            propagate()

        # Dessiner la grille
        for x in range(OUTPUT_WIDTH):
            for y in range(OUTPUT_HEIGHT):
                if is_collapsed(x, y):
                    tile_id = grid[x][y][0]
                    screen.blit(tile_surfaces[tile_id], (x*TILE_SIZE, y*TILE_SIZE))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# test3()


def test4():
    import pygame
    import math
    import random

    # Initialisation de Pygame
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulation Spatiale")

    # Couleurs
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    YELLOW = (255, 255, 0)
    PLANET_COLORS = [(100, 149, 237), (144, 238, 144), (255, 105, 180), (255, 165, 0)]

    # Paramètres du système
    FPS = 60
    G = 0.5  # Constante gravitationnelle artificielle

    class Planet:
        def __init__(self, x, y, radius, color, mass):
            self.x = x
            self.y = y
            self.radius = radius
            self.color = color
            self.mass = mass
            self.vx = random.uniform(-2, 2)
            self.vy = random.uniform(-2, 2)
            self.trail = []

        def update(self, star_x, star_y):
            # Calcul de la gravité
            dx = star_x - self.x
            dy = star_y - self.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance == 0:
                distance = 0.1
            force = G * self.mass / (distance**2)
            angle = math.atan2(dy, dx)
            self.vx += force * math.cos(angle)
            self.vy += force * math.sin(angle)

            # Mise à jour de la position
            self.x += self.vx
            self.y += self.vy

            # Sauvegarde du trail
            self.trail.append((self.x, self.y))
            if len(self.trail) > 50:
                self.trail.pop(0)

        def draw(self, win):
            # Dessin du trail
            for pos in self.trail:
                pygame.draw.circle(win, self.color, (int(pos[0]), int(pos[1])), 2)
            # Dessin de la planète
            pygame.draw.circle(win, self.color, (int(self.x), int(self.y)), self.radius)

    
    clock = pygame.time.Clock()
    run = True

    # Position de l'étoile
    star_x, star_y = WIDTH // 2, HEIGHT // 2
    star_radius = 30

    # Création de planètes
    planets = []
    for i in range(5):
        p = Planet(random.randint(100, WIDTH-100),
                random.randint(100, HEIGHT-100),
                random.randint(5, 10),
                random.choice(PLANET_COLORS),
                random.randint(5, 20))
        planets.append(p)

    while run:
        clock.tick(FPS)
        WIN.fill(BLACK)

        # Gestion des événements
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Dessin de l'étoile
        pygame.draw.circle(WIN, YELLOW, (star_x, star_y), star_radius)

        # Mise à jour et dessin des planètes
        for planet in planets:
            planet.update(star_x, star_y)
            planet.draw(WIN)

        pygame.display.update()

    pygame.quit()

# test4()


def test5():
    import pygame
    import random
    import math

    # Initialisation de Pygame
    pygame.init()

    # Taille de la fenêtre
    WINDOW_SIZE = 600
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Triangles rebondissants")

    # Paramètres
    NUM_TRIANGLES = 4
    SPEED = 3
    GROWTH_RATE = 1.02  # Chaque collision augmente la taille de 2%
    FPS = 60

    # Classe Triangle
    class Triangle:
        def __init__(self):
            self.size = 50
            self.x = random.randint(self.size, WINDOW_SIZE - self.size)
            self.y = random.randint(self.size, WINDOW_SIZE - self.size)
            self.vx = random.choice([-SPEED, SPEED])
            self.vy = random.choice([-SPEED, SPEED])
            self.color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
        
        def get_points(self):
            # Triangle équilatéral
            height = self.size * math.sqrt(3) / 2
            return [(self.x, self.y - 2/3*height),
                    (self.x - self.size/2, self.y + 1/3*height),
                    (self.x + self.size/2, self.y + 1/3*height)]
        
        def move(self):
            self.x += self.vx
            self.y += self.vy
            
            # Collision avec les bords
            if self.x - self.size/2 <= 0 or self.x + self.size/2 >= WINDOW_SIZE:
                self.vx *= -1
                self.size *= GROWTH_RATE
            if self.y - self.size/2 <= 0 or self.y + self.size/2 >= WINDOW_SIZE:
                self.vy *= -1
                self.size *= GROWTH_RATE
        
        def draw(self, surface):
            pygame.draw.polygon(surface, self.color, self.get_points())

    def check_collision(t1, t2):
        # Collision simple basée sur la distance entre centres
        dx = t1.x - t2.x
        dy = t1.y - t2.y
        distance = math.hypot(dx, dy)
        if distance < (t1.size + t2.size)/2:
            t1.vx *= -1
            t1.vy *= -1
            t2.vx *= -1
            t2.vy *= -1
            t1.size *= GROWTH_RATE
            t2.size *= GROWTH_RATE

    # Création des triangles
    triangles = [Triangle() for _ in range(NUM_TRIANGLES)]

    # Boucle principale
    clock = pygame.time.Clock()
    running = True
    while running:
        screen.fill((30, 30, 30))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Déplacement et dessin
        for tri in triangles:
            tri.move()
            tri.draw(screen)
        
        # Vérification collisions entre triangles
        for i in range(NUM_TRIANGLES):
            for j in range(i+1, NUM_TRIANGLES):
                check_collision(triangles[i], triangles[j])
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()



# test5()


def test6(width, height, output_file):
    import pygame
    import random
    import math
    import numpy as np
    import imageio

    # Initialisation de Pygame
    pygame.init()
    pygame.display.set_caption("🎄 Simulation de Noël – neige sur une colline 🎅")

    WIDTH, HEIGHT = width, height
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # 🎥 Writer vidéo imageio (mp4, 60 FPS)
    writer = imageio.get_writer(output_file, fps=60, codec='libx264', quality=8)

    # Couleurs
    WHITE = (255, 255, 255)
    DARK_BLUE = (0, 128, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)

    LINE_Y = 100
    ACCUMULATION_RATE = 4  

    base_height = [int(10 * math.sin(x / 100) + 50) for x in range(WIDTH)]
    snow_height = [base_height[x] for x in range(WIDTH)]

    class Snowflake:
        def __init__(self):
            self.x = random.randint(0, WIDTH - 1)
            self.y = random.randint(-HEIGHT, 0)
            self.radius = random.randint(2, 4)
            self.speed = random.uniform(1, 3)
            self.angle = random.uniform(0, math.pi * 2)
            self.is_black = random.random() < 0.025  

        def fall(self):
            self.y += self.speed
            self.x += math.sin(self.angle) * 0.5
            if self.x < 0: self.x += WIDTH
            elif self.x >= WIDTH: self.x -= WIDTH

        def draw(self, screen):
            color = BLACK if self.is_black else WHITE
            pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)

    snowflakes = [Snowflake() for _ in range(250)]
    running = True
    clock = pygame.time.Clock()
    snow_reached_line = False

    while running:
        screen.fill(DARK_BLUE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for x in range(WIDTH):
            ground_y = HEIGHT - snow_height[x]
            pygame.draw.line(screen, WHITE, (x, HEIGHT), (x, ground_y))

        pygame.draw.line(screen, GREEN, (0, LINE_Y), (WIDTH, LINE_Y), 3)

        for x in range(WIDTH):
            if HEIGHT - snow_height[x] <= LINE_Y:
                snow_reached_line = True
                break

        if snow_reached_line:
            font = pygame.font.SysFont("Comic Sans MS", 48, bold=True)
            text = font.render(" La neige a atteint la ligne !", True, GREEN)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 40))

        for flake in snowflakes:
            flake.fall()
            xi = int(flake.x)
            if 0 <= xi < WIDTH:
                ground_y = HEIGHT - snow_height[xi]
                if flake.y >= ground_y:
                    if flake.is_black:
                        for dx in range(-10, 11):
                            xj = xi + dx
                            if 0 <= xj < WIDTH:
                                snow_height[xj] = max(base_height[xj], snow_height[xj] - random.randint(5, 10))
                    else:
                        snow_height[xi] = min(HEIGHT, snow_height[xi] + ACCUMULATION_RATE)
                        if xi > 0: snow_height[xi - 1] = max(snow_height[xi - 1], snow_height[xi] - ACCUMULATION_RATE // 2)
                        if xi < WIDTH - 1: snow_height[xi + 1] = max(snow_height[xi + 1], snow_height[xi] - ACCUMULATION_RATE // 2)

                    snowflakes.remove(flake)
                    snowflakes.append(Snowflake())
                    continue

            flake.draw(screen)

        pygame.display.flip()

        # 🎥 Capture frame → numpy → enregistrement
        frame = pygame.surfarray.array3d(screen)       # (W,H,3)
        frame = np.transpose(frame, (1, 0, 2))         # Correction orientation
        writer.append_data(frame)

        if snow_reached_line:
            pygame.time.wait(3000)
            break

        clock.tick(60)

    writer.close()
    pygame.quit()


def main3():
    for format in ["h","v"]:
        if format == "h":
            width, height = 1080, 720
            output_dir = "../../Reddit/Simulations/Christmas/ChristmasHill/Horizontal/sim2.mp4"
        else:
            width, height = 720, 1080
            output_dir = "../../Reddit/Simulations/Christmas/ChristmasHill/Vertical/sim2.mp4"
        
        test6(width, height, output_dir)

main3()