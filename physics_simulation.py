import pygame
import random
import math
import time
from dataclasses import dataclass

from typing import List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import copy 

@dataclass(frozen=True)
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    mass: float

class Profiler:
    def __init__(self):
        self.frame_times = []
        self.collision_counts = []
        self._start_time = None
        self._collision_count = 0
        
    def start_frame(self):
        self._start_time = time.time()
        self._collision_count = 0
        
    def end_frame(self):
        if self._start_time is not None:
            frame_time = time.time() - self._start_time
            self.frame_times.append(frame_time)
            self.collision_counts.append(self._collision_count)
            
    def record_collision(self):
        self._collision_count += 1
        
    def get_stats(self) -> dict:
        if not self.frame_times:
            return {"avg_frame_time": 0, "avg_collisions": 0}
            
        # Keep only the last 100 frames for rolling average
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
            self.collision_counts = self.collision_counts[-100:]
            
        return {
            "avg_frame_time": sum(self.frame_times) / len(self.frame_times),
            "avg_collisions": sum(self.collision_counts) / len(self.collision_counts),
            "total_frames": len(self.frame_times)
        }
        
    def print_stats(self):
        stats = self.get_stats()
        print(f"Avg frame time: {stats['avg_frame_time']*1000:.2f} ms, "
              f"Avg collisions/frame: {stats['avg_collisions']:.2f}")


class ParticleSystem:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.profiler = Profiler()
        
    def create_particles(self, n: int):
        self.particles = [
            Particle(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                vx=random.uniform(-100, 100),
                vy=random.uniform(-100, 100),
                radius=random.uniform(2, 2),
                mass=math.pi * (random.uniform(2, 14) ** 2)
            )
            for _ in range(n)
        ]
        return self.particles
        
    def update(self, dt: float):
        self.profiler.start_frame()
        
        # Update positions
        self.particles = [self._update_position(p, dt) for p in self.particles]
        
        # Handle wall collisions
        self.particles = [self._handle_wall_collision(p) for p in self.particles]
        
        # Handle particle collisions
        self.particles = self._sweep_and_prune()
        
        self.profiler.end_frame()
        return self.particles
        
    def _update_position(self, p: Particle, dt: float) -> Particle:
        return Particle(
            x=p.x + p.vx * dt,
            y=p.y + p.vy * dt,
            vx=p.vx,
            vy=p.vy,
            radius=p.radius,
            mass=p.mass
        )
        
    def _handle_wall_collision(self, p: Particle) -> Particle:
        vx, vy, x, y = p.vx, p.vy, p.x, p.y

        if x - p.radius < 0 or x + p.radius > self.width:
            vx *= -1
            x = max(p.radius, min(x, self.width - p.radius))
        
        if y - p.radius < 0 or y + p.radius > self.height:
            vy *= -1
            y = max(p.radius, min(y, self.height - p.radius))
        
        return Particle(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            radius=p.radius,
            mass=p.mass
        )
        
    def _handle_particle_collision(self, p1: Particle, p2: Particle) -> Tuple[Particle, Particle]:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist = math.hypot(dx, dy)

        min_dist = p1.radius + p2.radius
        if dist >= min_dist or dist == 0:
            return p1, p2
            
        # Collision detected
        self.profiler.record_collision()
        
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

        # Resolve position overlap
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
        
    def _sweep_and_prune(self) -> List[Particle]:
        particles = self.particles
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
                    updated[idx], updated[j] = self._handle_particle_collision(updated[idx], updated[j])
                active.add(idx)
            else:
                active.remove(idx)

        return updated
        
    def get_profiler_stats(self):
        return self.profiler.get_stats()

class MultithreadedParticleSystem(ParticleSystem):
    """
    A particle system that uses multiple threads for simulation.
    Partitions particles by x-coordinate and processes each partition in a separate thread.
    """
    def __init__(self, width: int, height: int, num_threads: int = 8):
        super().__init__(width, height)
        self.num_threads = num_threads
        self.lock = threading.Lock()  # For thread-safe profiler access
        
    def update(self, dt: float):
        """Update all particles using multiple threads"""
        self.profiler.start_frame()

        # Partition particles by X-axis with overlap handling
        partitions, boundary_map = self._partition_particles_with_boundaries(self.particles, self.num_threads)

        # Process particles in threads
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self._thread_worker, part, dt, i) for i, part in enumerate(partitions)]
            results = [f.result() for f in futures]

        # Process boundary particles that might interact across partitions
        self._handle_boundary_collisions(results, boundary_map)
        
        # Merge updated local particles from threads
        self.particles = [p for part in results for p in part]
        
        self.profiler.end_frame()
        return self.particles
        
    def _partition_particles_with_boundaries(self, particles: List[Particle], num_parts: int) -> Tuple[List[List[Particle]], dict]:
        """
        Partitions particles by X-axis and identifies particles at partition boundaries.
        
        Returns:
            Tuple containing:
            - List of particle partitions
            - Dictionary mapping original indices to partition indices for boundary particles
        """
        partitions = [[] for _ in range(num_parts)]
        boundary_map = {}  # Maps (partition_idx, local_idx) to list of (neighbor_partition, particle) tuples
        
        x_slice_width = self.width / num_parts
        
        # First pass: assign particles to primary partitions
        for p_idx, p in enumerate(particles):
            partition_idx = int(p.x / x_slice_width)
            partition_idx = max(0, min(num_parts - 1, partition_idx))
            
            # Add to primary partition
            local_idx = len(partitions[partition_idx])
            partitions[partition_idx].append(p)
            
            # Check if the particle is at a boundary (within radius distance)
            is_left_boundary = partition_idx > 0 and p.x - p.radius <= partition_idx * x_slice_width
            is_right_boundary = partition_idx < num_parts - 1 and p.x + p.radius >= (partition_idx + 1) * x_slice_width
            
            if is_left_boundary or is_right_boundary:
                boundary_map[(partition_idx, local_idx)] = []
                
                # Add reference to left neighbor if needed
                if is_left_boundary:
                    boundary_map[(partition_idx, local_idx)].append((partition_idx - 1, p))
                    
                # Add reference to right neighbor if needed
                if is_right_boundary:
                    boundary_map[(partition_idx, local_idx)].append((partition_idx + 1, p))
        
        return partitions, boundary_map
                
    def _handle_boundary_collisions(self, updated_partitions: List[List[Particle]], boundary_map: dict):
        """
        Handles collisions between particles at partition boundaries.
        
        Args:
            updated_partitions: List of updated particle lists from each thread
            boundary_map: Dictionary mapping particle positions to boundary information
        """
        # Process all boundary particles
        for (part_idx, local_idx), neighbors in boundary_map.items():
            if part_idx >= len(updated_partitions) or local_idx >= len(updated_partitions[part_idx]):
                continue  # Skip if indices are no longer valid (particle might have moved)
                
            # Get the updated particle from its primary partition
            p1 = updated_partitions[part_idx][local_idx]
            
            # Check for collisions with particles in neighboring partitions
            for neighbor_part_idx, _ in neighbors:
                if neighbor_part_idx >= len(updated_partitions):
                    continue
                    
                # Perform sweep and prune for this particle against the neighbor partition
                for j, p2 in enumerate(updated_partitions[neighbor_part_idx]):
                    # Check if particles could collide (simple distance check)
                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    dist = math.hypot(dx, dy)
                    min_dist = p1.radius + p2.radius
                    
                    if dist < min_dist:  # Collision detected
                        # Update both particles
                        updated_p1, updated_p2 = self._handle_particle_collision(p1, p2)
                        updated_partitions[part_idx][local_idx] = updated_p1
                        updated_partitions[neighbor_part_idx][j] = updated_p2
                        p1 = updated_p1  # Update p1 for subsequent checks

    def _partition_particles(self, particles: List[Particle], num_parts: int) -> List[List[Particle]]:
        """Original partitioning method (for backward compatibility)"""
        partitions = [[] for _ in range(num_parts)]
        x_slice_width = self.width / num_parts
        for p in particles:
            idx = int(p.x / x_slice_width)
            idx = max(0, min(num_parts - 1, idx))
            partitions[idx].append(p)
        return partitions
        
    def _thread_worker(self, local_particles: List[Particle], dt: float, thread_id: int) -> List[Particle]:
        """Worker function that processes a subset of particles in a thread"""
        # Deep copy to avoid mutating shared data
        particles = copy.deepcopy(local_particles)

        # Update positions
        updated = [self._update_position(p, dt) for p in particles]

        # Handle wall collisions
        updated = [self._handle_wall_collision(p) for p in updated]

        # Handle local particle collisions using sweep and prune
        updated = self._thread_sweep_and_prune(updated)

        return updated
        
    def _thread_sweep_and_prune(self, particles: List[Particle]) -> List[Particle]:
        """
        Implements sweep and prune algorithm for efficient collision detection within a thread.
        """
        n = len(particles)
        if n < 2:
            return particles[:]

        # Create indexed particles and endpoints for sweep and prune
        indexed_particles = list(enumerate(particles))
        endpoints = [(p.x - p.radius, True, i) for i, p in indexed_particles] + \
                    [(p.x + p.radius, False, i) for i, p in indexed_particles]
        endpoints.sort()  # Sort by x-coordinate

        active = set()  # Set of active particle indices
        updated = particles[:]  # Copy particles for updating

        for _, is_start, idx in endpoints:
            if is_start:
                # When a particle starts, check collision with all active particles
                for j in active:
                    updated[idx], updated[j] = self._handle_particle_collision(updated[idx], updated[j])
                active.add(idx)
            else:
                # When a particle ends, remove it from the active set
                active.remove(idx)

        return updated
        
    def _handle_particle_collision(self, p1: Particle, p2: Particle) -> Tuple[Particle, Particle]:
        """Handle collision between two particles with thread-safe profiling"""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist = math.hypot(dx, dy)
        min_dist = p1.radius + p2.radius

        if dist >= min_dist or dist == 0:
            return p1, p2

        with self.lock:
            self.profiler.record_collision()

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


class ParticleRenderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.default_color = (255, 255, 255)
        self.background_color = (0, 0, 0)
        self.font = None
        self._init_font()
        
    def _init_font(self):
        try:
            self.font = pygame.font.SysFont('Arial', 14)
        except:
            pass  # Font not available
        
    def render(self, particles: List[Particle], stats: Optional[dict] = None):
        self.screen.fill(self.background_color)
        
        # Draw all particles
        for p in particles:
            pygame.draw.circle(
                self.screen, 
                self.default_color, 
                (int(p.x), int(p.y)), 
                int(p.radius)
            )
            
        # Display stats if available
        if stats and self.font:
            fps = 1.0 / stats["avg_frame_time"] if stats["avg_frame_time"] > 0 else 0
            stats_text = f"FPS: {fps:.1f} | Collisions/frame: {stats['avg_collisions']:.1f}"
            text_surface = self.font.render(stats_text, True, (255, 255, 0))
            self.screen.blit(text_surface, (10, 10))
            
        pygame.display.flip()


def run_pygame_simulation():
    """
    Main entry point for the simulation.
    Sets up the simulation components and runs the main loop.
    """
    # Initialize pygame
    pygame.init()
    width, height = 1400, 1000
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Particle Simulation")
    clock = pygame.time.Clock()
    # Create the modular components
    # Choose between single-threaded or multi-threaded system
    use_threading = True  # Set to False to use single-threaded version
    
    if use_threading:
        # Use all available CPU cores minus 1 (to leave one for UI)
        import os
        num_threads = 8 #max(1, os.cpu_count() - 1) if os.cpu_count() else 4
        particle_system = MultithreadedParticleSystem(width, height, num_threads=num_threads)
        print(f"Using multithreaded system with {num_threads} threads")
    else:
        particle_system = ParticleSystem(width, height)
        print("Using single-threaded system")
        
    particle_system.create_particles(10000)
    renderer = ParticleRenderer(screen)
    
    # Main loop variables
    running = True
    show_stats = True
    stats_update_timer = 0
    
    while running:
        # Handle timing
        dt = clock.tick(60) / 1000.0  # Time in seconds since last frame
        stats_update_timer += dt
        
        # Handle input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Toggle stats display
                    show_stats = not show_stats
                elif event.key == pygame.K_r:  # Reset simulation
                    particle_system.create_particles(1000)
        
        # Update simulation
        particle_system.update(dt)
        
        # Render
        if show_stats:
            stats = particle_system.get_profiler_stats()
            renderer.render(particle_system.particles, stats)
            
            # Print stats every second
            if stats_update_timer >= 1.0:
                particle_system.profiler.print_stats()
                stats_update_timer = 0
        else:
            renderer.render(particle_system.particles)
    
    pygame.quit()


if __name__ == "__main__":
    run_pygame_simulation()