import pygame
import numpy as np
import math

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Light Simulation")
clock = pygame.time.Clock()

# --- Settings Variables ---
scope_to_shoot = (0, 360)       # Angular range (in degrees) for ray emission
NumberOfRays = 360               # Number of initial rays from the light source
ShowOnlyHitRay = 1              # If True, only draw rays that hit an object
max_reflections = 2             # Maximum number of reflections per ray

# --- Primary Mirror Settings ---
# mirror_type: 0 = plane, 1 = concave, 2 = convex
mirror_type = 2

# For mirror_type 0 (plane mirror), we store center, length, and rotation.
mirror_center = np.array([500.0, 300.0])
orig_p1 = np.array([400.0, 250.0])
orig_p2 = np.array([600.0, 350.0])
mirror_length = np.linalg.norm(orig_p2 - orig_p1)
mirror_rotation = math.atan2(orig_p2[1] - orig_p1[1], orig_p2[0] - orig_p1[0])

# For mirror_type 1 and 2 (circular mirrors)
circle_center = np.array([500.0, 300.0])
circle_radius = 100.0

mirror_dragging = False
mirror_selected = False
mirror_offset = np.array([0.0, 0.0])

# --- Light Source ---
light_pos = np.array([150.0, 300.0])
light_radius = 10
light_dragging = False

# --- Scene Objects ---
# Created via keys:
# q: circle, w: square, e: rectangle, r: glassbox (for refraction)
scene_objects = []
selected_object = None   # last selected object (for dragging/rotation)
dragging_object = None
object_drag_offset = np.array([0.0, 0.0])

# --- Utility Functions ---

def get_plane_mirror_endpoints(center, length, rotation):
    half = length / 2.0
    direction = np.array([math.cos(rotation), math.sin(rotation)])
    return center + half * direction, center - half * direction

def point_line_distance(pt, a, b):
    ap = pt - a
    ab = b - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
    closest = a + t * ab
    return np.linalg.norm(pt - closest)

def ray_line_intersection(ray_origin, ray_dir, p1, p2):
    v1 = ray_origin - p1
    v2 = p2 - p1
    v3 = np.array([-ray_dir[1], ray_dir[0]])
    dot = np.dot(v2, v3)
    if abs(dot) < 1e-6:
        return None, None, None
    t1 = np.cross(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot
    if t1 >= 1e-6 and 0 <= t2 <= 1:
        hit = ray_origin + t1 * ray_dir
        normal = np.array([-v2[1], v2[0]])
        normal = normal / np.linalg.norm(normal)
        if np.dot(ray_dir, normal) > 0:
            normal = -normal
        return hit, normal, t1
    return None, None, None

def ray_circle_intersection(ray_origin, ray_dir, center, radius):
    oc = ray_origin - center
    a = np.dot(ray_dir, ray_dir)
    b = 2 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - radius**2
    disc = b*b - 4*a*c
    if disc < 0:
        return None, None, None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2*a)
    t2 = (-b + sqrt_disc) / (2*a)
    t = None
    if t1 >= 1e-6 and t2 >= 1e-6:
        t = min(t1, t2)
    elif t1 >= 1e-6:
        t = t1
    elif t2 >= 1e-6:
        t = t2
    if t is None:
        return None, None, None
    hit = ray_origin + t * ray_dir
    normal = hit - center
    normal = normal / np.linalg.norm(normal)
    return hit, normal, t

def compute_rectangle_vertices(center, width, height, rotation):
    hw, hh = width/2, height/2
    corners = np.array([[-hw, -hh],
                        [ hw, -hh],
                        [ hw,  hh],
                        [-hw,  hh]])
    rot_matrix = np.array([[math.cos(rotation), -math.sin(rotation)],
                           [math.sin(rotation),  math.cos(rotation)]])
    return np.dot(corners, rot_matrix.T) + center

def ray_polygon_intersection(ray_origin, ray_dir, vertices):
    closest_hit = None
    closest_normal = None
    min_t = float('inf')
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i+1)%n]
        hit, normal, t = ray_line_intersection(ray_origin, ray_dir, p1, p2)
        if hit is not None and t < min_t:
            min_t = t
            closest_hit = hit
            closest_normal = normal
    if closest_hit is not None:
        return closest_hit, closest_normal, min_t
    return None, None, None

def point_in_polygon(point, vertices):
    x, y = point
    inside = False
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1)%n]
        if ((y1 > y) != (y2 > y)) and (x < (x2-x1) * (y-y1) / (y2-y1) + x1):
            inside = not inside
    return inside

def refract(I, N, n1, n2):
    cos_i = -np.dot(N, I)
    ratio = n1 / n2
    sin_t2 = ratio**2 * (1 - cos_i**2)
    if sin_t2 > 1:
        return None
    cos_t = math.sqrt(1 - sin_t2)
    return ratio * I + (ratio * cos_i - cos_t) * N

def intersect_rays(ray1, ray2):
    # Each ray is represented as a tuple (origin, direction)
    p1, d1 = ray1
    p2, d2 = ray2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    b = p2 - p1
    t_vals = np.linalg.solve(A, b)
    t1, t2 = t_vals
    if t1 >= 0 and t2 >= 0:
        return p1 + t1 * d1
    return None

# Pre-create a surface for the image marker (semi-transparent red)
img_marker_size = 16
img_marker = pygame.Surface((img_marker_size, img_marker_size), pygame.SRCALPHA)
img_marker.fill((255, 0, 0, 128))  # red with 50% opacity

# --- Main Loop ---
running = True
while running:
    # Clear previous frame's reflected rays and reset hit counts.
    for obj in scene_objects:
        obj['reflected_rays'] = []
        obj['hit_count'] = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # --- KEY EVENTS ---
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0:
                mirror_type = 0
            elif event.key == pygame.K_1:
                mirror_type = 1
            elif event.key == pygame.K_2:
                mirror_type = 2

            if event.key == pygame.K_q:
                pos = np.array(pygame.mouse.get_pos(), dtype=float)
                scene_objects.append({'type': 'circle', 'center': pos, 'radius': 40, 'rotation': 0, 'hit_count': 0})
                selected_object = scene_objects[-1]
            elif event.key == pygame.K_w:
                pos = np.array(pygame.mouse.get_pos(), dtype=float)
                scene_objects.append({'type': 'square', 'center': pos, 'size': 80, 'rotation': 0, 'hit_count': 0})
                selected_object = scene_objects[-1]
            elif event.key == pygame.K_e:
                pos = np.array(pygame.mouse.get_pos(), dtype=float)
                scene_objects.append({'type': 'rectangle', 'center': pos, 'width': 120, 'height': 60, 'rotation': 0, 'hit_count': 0})
                selected_object = scene_objects[-1]
            elif event.key == pygame.K_r:
                pos = np.array(pygame.mouse.get_pos(), dtype=float)
                scene_objects.append({'type': 'glassbox', 'center': pos, 'width': 120, 'height': 60, 'rotation': 0, 'ior': 1.5, 'hit_count': 0})
                selected_object = scene_objects[-1]

            if event.key == pygame.K_LEFT:
                if mirror_selected and mirror_type == 0:
                    mirror_rotation -= math.radians(5)
                elif selected_object is not None:
                    selected_object['rotation'] -= math.radians(5)
            elif event.key == pygame.K_RIGHT:
                if mirror_selected and mirror_type == 0:
                    mirror_rotation += math.radians(5)
                elif selected_object is not None:
                    selected_object['rotation'] += math.radians(5)
            if event.key == pygame.K_s:
                ShowOnlyHitRay = not ShowOnlyHitRay

        # --- MOUSE EVENTS ---
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
            if np.linalg.norm(mouse_pos - light_pos) < light_radius + 5:
                light_dragging = True
            if mirror_type == 0:
                p1, p2 = get_plane_mirror_endpoints(mirror_center, mirror_length, mirror_rotation)
                if point_line_distance(mouse_pos, p1, p2) < 10:
                    mirror_dragging = True
                    mirror_selected = True
                    mirror_offset = mouse_pos - mirror_center
            else:
                if abs(np.linalg.norm(mouse_pos - circle_center) - circle_radius) < 10:
                    mirror_dragging = True
                    mirror_selected = True
                    mirror_offset = mouse_pos - circle_center
            for obj in reversed(scene_objects):
                if obj['type'] == 'circle':
                    if np.linalg.norm(mouse_pos - obj['center']) < obj['radius']:
                        dragging_object = obj
                        selected_object = obj
                        mirror_selected = False
                        object_drag_offset = mouse_pos - obj['center']
                        break
                else:
                    if obj['type'] == 'square':
                        vertices = compute_rectangle_vertices(obj['center'], obj['size'], obj['size'], obj['rotation'])
                    elif obj['type'] in ['rectangle', 'glassbox']:
                        vertices = compute_rectangle_vertices(obj['center'], obj['width'], obj['height'], obj['rotation'])
                    if point_in_polygon(mouse_pos, vertices):
                        dragging_object = obj
                        selected_object = obj
                        mirror_selected = False
                        object_drag_offset = mouse_pos - obj['center']
                        break

        if event.type == pygame.MOUSEBUTTONUP:
            light_dragging = False
            mirror_dragging = False
            dragging_object = None

        if event.type == pygame.MOUSEMOTION:
            mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
            if light_dragging:
                light_pos = mouse_pos
            if mirror_dragging:
                if mirror_type == 0:
                    mirror_center = mouse_pos - mirror_offset
                else:
                    circle_center = mouse_pos - mirror_offset
            if dragging_object is not None:
                dragging_object['center'] = mouse_pos - object_drag_offset

    # --- Drawing Section ---
    screen.fill((30, 30, 30))
    mirror_color = (200, 200, 200)
    if mirror_type == 0:
        p1, p2 = get_plane_mirror_endpoints(mirror_center, mirror_length, mirror_rotation)
        pygame.draw.line(screen, mirror_color, p1.astype(int), p2.astype(int), 4)
    else:
        pygame.draw.circle(screen, mirror_color, circle_center.astype(int), int(circle_radius), 4)
    for obj in scene_objects:
        if obj['type'] == 'circle':
            col = (0, 150, 150) if obj.get('hit_count', 0) >= 2 else (150, 150, 150)
            pygame.draw.circle(screen, col, obj['center'].astype(int), int(obj['radius']))
        elif obj['type'] == 'square':
            vertices = compute_rectangle_vertices(obj['center'], obj['size'], obj['size'], obj['rotation'])
            col = (0, 150, 150) if obj.get('hit_count', 0) >= 2 else (150, 150, 150)
            pygame.draw.polygon(screen, col, vertices)
        elif obj['type'] == 'rectangle':
            vertices = compute_rectangle_vertices(obj['center'], obj['width'], obj['height'], obj['rotation'])
            col = (0, 150, 150) if obj.get('hit_count', 0) >= 2 else (150, 150, 150)
            pygame.draw.polygon(screen, col, vertices)
        elif obj['type'] == 'glassbox':
            vertices = compute_rectangle_vertices(obj['center'], obj['width'], obj['height'], obj['rotation'])
            col = (150, 150, 250) if obj.get('hit_count', 0) >= 2 else (200, 200, 255)
            pygame.draw.polygon(screen, col, vertices)
    pygame.draw.circle(screen, (255, 255, 0), light_pos.astype(int), light_radius)

    # --- Ray Casting with Multi-Reflection ---
    for i in range(NumberOfRays):
        angle = math.radians(scope_to_shoot[0] + (scope_to_shoot[1] - scope_to_shoot[0]) * i / NumberOfRays)
        current_origin = light_pos.copy()
        current_dir = np.array([math.cos(angle), math.sin(angle)])
        current_dir /= np.linalg.norm(current_dir)
        reflection_count = 0

        while reflection_count < max_reflections:
            closest_hit = None
            closest_normal = None
            min_t = float('inf')
            hit_obj_type = None
            hit_obj = None

            if mirror_type == 0:
                p1, p2 = get_plane_mirror_endpoints(mirror_center, mirror_length, mirror_rotation)
                hit, normal, t = ray_line_intersection(current_origin, current_dir, p1, p2)
            else:
                hit, normal, t = ray_circle_intersection(current_origin, current_dir, circle_center, circle_radius)
            if hit is not None and t < min_t:
                min_t = t
                closest_hit = hit
                closest_normal = normal
                hit_obj_type = 'primary_mirror'

            for obj in scene_objects:
                if obj['type'] == 'circle':
                    hit_c, normal_c, t_c = ray_circle_intersection(current_origin, current_dir, obj['center'], obj['radius'])
                    if hit_c is not None and t_c < min_t:
                        min_t = t_c
                        closest_hit = hit_c
                        closest_normal = normal_c
                        hit_obj_type = obj['type']
                        hit_obj = obj
                else:
                    if obj['type'] == 'square':
                        vertices = compute_rectangle_vertices(obj['center'], obj['size'], obj['size'], obj['rotation'])
                    elif obj['type'] in ['rectangle', 'glassbox']:
                        vertices = compute_rectangle_vertices(obj['center'], obj['width'], obj['height'], obj['rotation'])
                    hit_p, normal_p, t_p = ray_polygon_intersection(current_origin, current_dir, vertices)
                    if hit_p is not None and t_p < min_t:
                        min_t = t_p
                        closest_hit = hit_p
                        closest_normal = normal_p
                        hit_obj_type = obj['type']
                        hit_obj = obj

            if closest_hit is None:
                # If ShowOnlyHitRay is enabled, do not draw the segment that misses.
                if not ShowOnlyHitRay:
                    end_point = current_origin + current_dir * 1000
                    pygame.draw.line(screen, (255, 255, 255), current_origin.astype(int), end_point.astype(int), 1)
                break

            # Draw the segment from current_origin to the hit if ShowOnlyHitRay is off or hit exists.
            pygame.draw.line(screen, (255, 255, 255), current_origin.astype(int), closest_hit.astype(int), 1)

            # Compute secondary direction.
            if hit_obj is not None and hit_obj.get('type') == 'glassbox':
                n1 = 1.0
                n2 = hit_obj.get('ior', 1.5)
                refracted = refract(current_dir, closest_normal, n1, n2)
                if refracted is not None:
                    sec_dir = refracted / np.linalg.norm(refracted)
                    ray_color = (0, 255, 0)
                else:
                    sec_dir = (current_dir - 2 * np.dot(current_dir, closest_normal) * closest_normal)
                    sec_dir /= np.linalg.norm(sec_dir)
                    ray_color = (0, 255, 255)
            else:
                sec_dir = (current_dir - 2 * np.dot(current_dir, closest_normal) * closest_normal)
                sec_dir /= np.linalg.norm(sec_dir)
                ray_color = (0, 255, 255)

            second_end = closest_hit + sec_dir * 300
            pygame.draw.line(screen, ray_color, closest_hit.astype(int), second_end.astype(int), 1)

            if hit_obj is not None:
                hit_obj['hit_count'] += 1
                hit_obj.setdefault('reflected_rays', []).append((closest_hit, sec_dir))

            epsilon = 1e-3
            current_origin = closest_hit + sec_dir * epsilon
            current_dir = sec_dir
            reflection_count += 1

    # --- Image Formation ---
    # For each scene object (except primary mirror), if at least two secondary rays are recorded,
    # compute pairwise intersections. Then, draw a semi-transparent red marker (via a Surface)
    # centered at the average intersection location.
    for obj in scene_objects:
        rays = obj.get('reflected_rays', [])
        if len(rays) >= 2:
            intersections = []
            for i in range(len(rays)):
                for j in range(i+1, len(rays)):
                    inter = intersect_rays(rays[i], rays[j])
                    if inter is not None:
                        intersections.append(inter)
            if intersections:
                avg_int = np.mean(intersections, axis=0)
                pos = (int(avg_int[0] - img_marker_size/2), int(avg_int[1] - img_marker_size/2))
                screen.blit(img_marker, pos)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
