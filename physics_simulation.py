import pygame
import random
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    mass: float

def update_position(p: Particle, dt: float):
    return Particle(
        x=p.x + p.vx * dt,
        y=p.y + p.vy * dt,
        vx=p.vx,
        vy=p.vy,
        radius=p.radius,
        mass=p.mass
    )

def handle_wall_collision(p: Particle, width: int, height: int):
    vx, vy, x, y = p.vx, p.vy, p.x, p.y

    if x - p.radius < 0 or x + p.radius > width:
        vx *= -1
        x = max(p.radius, min(x, width - p.radius))
    
    if y - p.radius < 0 or y + p.radius > height:
        vy *= -1
        y = max(p.radius, min(y, height - p.radius))
    
    return Particle(
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        radius=p.radius,
        mass=p.mass
    )
    
def handle_particle_collision(p1: Particle, p2: Particle):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dist = math.hypot(dx, dy)

    min_dist = p1.radius + p2.radius
    if dist >= min_dist or dist == 0:
        return p1, p2
    
    nx = dx / dist
    ny = dy / dist
    tx = -ny
    ty = nx

    dpTan1 = p1.vx * tx + p1.vy * ty
    dpTan2 = p2.vx * tx + p2.vy * ty
    dpNorm1 = p1.vx * nx + p1.vy * ny
    dpNorm2 = p2.vx * nx + p2.vy * ny

    m1, m2 = p1.mass, p2.mass

    v1n = (dpNorm1 * (m1 - m2) + 2 * m2 * dpNorm2) / (m1 + m2)
    v2n = (dpNorm2 * (m2 - m1) + 2 * m1 * dpNorm1) / (m1 + m2)

    vx1 = tx * dpTan1 + nx * v1n
    vy1 = ty * dpTan1 + ny * v1n
    vx2 = tx * dpTan2 + nx * v2n
    vy2 = ty * dpTan2 + ny * v2n

    overlap = min_dist - dist
    connection = overlap / (m1 + m2)
    x1 = p1.x - nx * connection * m2
    y1 = p1.y - ny * connection * m2
    x2 = p2.x + nx * connection * m1
    y2 = p2.y + ny * connection * m1

    return (
        Particle(x1, y1, vx1, vy1, p1.radius, p1.mass),
        Particle(x2, y2, vx2, vy2, p2.radius, p2.mass)
    )

def sweep_and_prune(particles: list[Particle]):
    n = len(particles)
    if n < 2:
        return particles[:]

    indexed_particles = list(enumerate(particles))
    endpoints = [(p.x - p.radius, True, i) for i, p in indexed_particles] + \
                [(p.x + p.radius, False, i) for i, p in indexed_particles]
    endpoints.sort()

    active = set()
    updated = particles[:]

    for _, is_start, idx in endpoints:
        if is_start:
            for j in active:
                updated[idx], updated[j] = handle_particle_collision(updated[idx], updated[j])
            active.add(idx)
        else:
            active.remove(idx)

    return updated
            

def simulate_steps(particles, dt, width, height):
    particles = [update_position(p, dt) for p in particles]
    particles = [handle_wall_collision(p, width, height) for p in particles]
    particles = sweep_and_prune(particles)
    return particles

def create_particles(n, width, height):
    return [
        Particle(
            x=random.uniform(0, width),
            y=random.uniform(0, height),
            vx=random.uniform(-100, 100),
            vy=random.uniform(-100, 100),
            radius=random.uniform(2, 5),
            mass=math.pi * (random.uniform(2, 14) ** 2)
        )
        for _ in range(n)
    ]

def run_pygame_simulation():
    pygame.init()
    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle Simulation")
    clock = pygame.time.Clock()

    particles = create_particles(1000, width, height)
    running = True

    while running:
        dt = clock.tick(60) / 1000.0  # Time in seconds since last frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        particles = simulate_steps(particles, dt, width, height)

        screen.fill((0, 0, 0))
        for p in particles:
            pygame.draw.circle(screen, (255, 255, 255), (int(p.x), int(p.y)), int(p.radius))
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    run_pygame_simulation()