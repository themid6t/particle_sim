import pygame
import random
import math
import time
from dataclasses import dataclass
import argparse
import psutil  # Import psutil for memory tracking

from typing import List, Tuple, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
import copy 
from benchmark import run_benchmark_simulation, run_extended_benchmark


# Particle parameters
VELOCITY_RANGE = (-100, 100)  # Range for particle velocity
RADIUS_RANGE = (2, 14)         # Range for particle radius
MASS_RADIUS_RANGE = (2, 8)   # Range for radius used in mass calculation

# Simulation parameters
SCREEN_WIDTH = 1440          # Screen width
SCREEN_HEIGHT = 800          # Screen height
FRAME_RATE = 60               # Frames per second

PARTICLE_COUNT = 1000  # Default number of particles in the simulation

# Additional global constants
NUM_THREADS = 8  # Number of threads for multithreaded simulation

# BACKGROUND_COLOR = (0, 0, 0)  # Screen background color
BACKGROUND_COLOR = (255, 255, 255)  # Screen background color
FONT_COLOR = (255, 255, 0)

if BACKGROUND_COLOR == (255, 255, 255):
    FONT_COLOR = (0, 0, 0)


@dataclass(frozen=True)
class Particle:
    """
    Represents a particle in the simulation.

    Attributes:
        x (float): X-coordinate of the particle.
        y (float): Y-coordinate of the particle.
        vx (float): Velocity of the particle along the X-axis.
        vy (float): Velocity of the particle along the Y-axis.
        radius (float): Radius of the particle.
        mass (float): Mass of the particle, calculated based on its radius.
        color (Tuple[int, int, int]): Color of the particle, determined by its mass.
    """
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    mass: float = 0  # Default value, will be calculated
    color: Tuple[int, int, int] = (255, 255, 255)  # Default value, will be calculated

    def __post_init__(self):
        object.__setattr__(self, 'mass', math.pi * min(self.radius, MASS_RADIUS_RANGE[1]) ** 2)  # Mass depends on radius
        max_mass = math.pi * MASS_RADIUS_RANGE[1] ** 2  # Max radius is 8
        mass_ratio = self.mass / max_mass # normalize mass to [0, 1]

        if mass_ratio <= 0.5:
            # Blue to Green
            t = mass_ratio / 0.5
            r = int(255 * (1 - t))
            g = int(255 * t)
            b = int(255 * (1 - t))
        else:
            # Green to Red
            t = (mass_ratio - 0.5) / 0.5
            r = int(255 * t)
            g = int(255 * (1 - t))
            b = 0

        color = (r, g, b)
        object.__setattr__(self, 'color', color)


class Profiler:
    """
    Tracks performance metrics for the simulation, such as frame times and collision counts.
    """
    def __init__(self):
        self.frame_times = []
        self.collision_counts = []
        self.memory_usages = []  # Track memory usage
        self._start_time = None
        self._collision_count = 0
        
    def start_frame(self):
        """
        Starts timing for a new frame.
        """
        self._start_time = time.time()
        self._collision_count = 0
        
    def end_frame(self):
        """
        Ends timing for the current frame and records statistics.
        """
        if self._start_time is not None:
            frame_time = time.time() - self._start_time
            self.frame_times.append(frame_time)
            self.collision_counts.append(self._collision_count)

            # Record overall memory usage for the entire simulation process
            # process = psutil.Process()
            # total_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            # self.memory_usages.append(total_memory)  # Track total memory usage

    def record_collision(self):
        """
        Records a collision event for the current frame.
        """
        self._collision_count += 1
        
    def get_stats(self) -> dict:
        """
        Returns performance statistics, including average frame time, collisions per frame, and memory usage.

        Returns:
            dict: A dictionary containing average frame time, average collisions, average memory usage, and total frames.
        """
        if not self.frame_times:
            return {"avg_frame_time": 0, "avg_collisions": 0, "avg_memory_usage": 0}
            
        # Keep only the last 100 frames for rolling average
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]
            self.collision_counts = self.collision_counts[-100:]
            self.memory_usages = self.memory_usages[-100:]

        return {
            "avg_frame_time": sum(self.frame_times) / len(self.frame_times),
            "avg_collisions": sum(self.collision_counts) / len(self.collision_counts),
            # "avg_memory_usage": sum(self.memory_usages) / len(self.memory_usages),
            "total_frames": len(self.frame_times)
        }
        
    def print_stats(self):
        stats = self.get_stats()
        print(f"Avg frame time: {stats['avg_frame_time']*1000:.2f} ms, "
              f"Avg collisions/frame: {stats['avg_collisions']:.2f}, ")
            #   f"Avg memory usage: {stats['avg_memory_usage']:.2f} MB")


class ParticleSystem:
    """
    Manages the simulation of particles, including their creation, updates, and collision handling.
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.profiler = Profiler()
        
    def create_particles(self, n: int):
        """
        Creates a specified number of particles with random properties.

        Args:
            n (int): Number of particles to create.

        Returns:
            List[Particle]: A list of created particles.
        """
        self.particles = [
            Particle(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                vx=random.uniform(*VELOCITY_RANGE),
                vy=random.uniform(*VELOCITY_RANGE),
                radius=random.uniform(*RADIUS_RANGE)
            )
            for _ in range(n)
        ]
        return self.particles
        
    def update(self, dt: float):
        """
        Updates the state of all particles in the system.

        Args:
            dt (float): Time step for the update.

        Returns:
            List[Particle]: The updated list of particles.
        """
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
        """
        Resolves a collision between two particles.

        Args:
            p1 (Particle): The first particle.
            p2 (Particle): The second particle.

        Returns:
            Tuple[Particle, Particle]: The updated states of the two particles after collision.
        """
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
        overlap = max(0, min_dist - dist)  # Ensure overlap is non-negative
        connection = overlap / (m1 + m2)
        x1 = p1.x - nx * connection * m2
        y1 = p1.y - ny * connection * m2
        x2 = p2.x + nx * connection * m1
        y2 = p2.y + ny * connection * m1

        return (
            Particle(x1, y1, vx1, vy1, p1.radius),
            Particle(x2, y2, vx2, vy2, p2.radius)
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
    Extends ParticleSystem to use multiple threads for simulation.
    Partitions particles by x-coordinate and processes each partition in a separate thread.
    """
    def __init__(self, width: int, height: int, num_threads: int = 8):
        """
        Initializes the multithreaded particle system.

        Args:
            width (int): Width of the simulation area.
            height (int): Height of the simulation area.
            num_threads (int): Number of threads to use for simulation.
        """
        super().__init__(width, height)
        self.num_threads = num_threads
        self.lock = threading.Lock()  # For thread-safe profiler access
        
    def update(self, dt: float):
        """
        Updates all particles using multiple threads.

        Args:
            dt (float): Time step for the update.

        Returns:
            List[Particle]: The updated list of particles.
        """
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

        Args:
            particles (List[Particle]): List of particles to partition.
            num_parts (int): Number of partitions.

        Returns:
            Tuple[List[List[Particle]], dict]:
                - List of particle partitions.
                - Dictionary mapping original indices to partition indices for boundary particles.
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
        Handles collisions between particles at partition boundaries using locks for thread safety.

        Args:
            updated_partitions (List[List[Particle]]): List of updated particle lists from each thread.
            boundary_map (dict): Dictionary mapping particle positions to boundary information.
        """
        locks = [threading.Lock() for _ in range(len(updated_partitions))]  # One lock per partition

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

                # Lock both partitions to ensure thread-safe access
                with locks[part_idx], locks[neighbor_part_idx]:
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

        Args:
            particles (List[Particle]): List of particles to process.

        Returns:
            List[Particle]: The updated list of particles after handling collisions.
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

        overlap = max(0, min_dist - dist)  # Ensure overlap is non-negative
        connection = overlap / (m1 + m2)
        x1 = p1.x - nx * connection * m2
        y1 = p1.y - ny * connection * m2
        x2 = p2.x + nx * connection * m1
        y2 = p2.y + ny * connection * m1

        return (
            Particle(x1, y1, vx1, vy1, p1.radius),
            Particle(x2, y2, vx2, vy2, p2.radius)
        )


class ParticleRenderer:
    """
    Handles rendering of particles and simulation statistics using Pygame.
    """
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.background_color = BACKGROUND_COLOR
        self.font = None
        self._init_font()

    def _init_font(self):
        try:
            self.font = pygame.font.SysFont('Arial', 14)
        except:
            pass  # Font not available

    def render(self, particles: List[Particle], stats: Optional[dict] = None, draw_boundaries: bool = False):
        """
        Renders particles and displays statistics on the screen.

        Args:
            particles (List[Particle]): List of particles to render.
            stats (Optional[dict]): Simulation statistics to display.
            draw_boundaries (bool): Whether to draw thread-separated boundaries.
            draw_buffers (bool): Whether to draw buffer regions around boundaries.
        """
        self.screen.fill(self.background_color)

        # Draw thread-separated boundaries if enabled
        if draw_boundaries:
            partition_width = SCREEN_WIDTH / NUM_THREADS
            for i in range(1, NUM_THREADS):
                x = int(i * partition_width)

                # Draw buffer regions
                buffer_color = (220, 220, 220)  # Light gray for buffer regions
                buffer_width = max(RADIUS_RANGE) * 2
                pygame.draw.rect(self.screen, buffer_color, (x - buffer_width, 0, buffer_width * 2, SCREEN_HEIGHT))

                pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, SCREEN_HEIGHT), 1)  # Light gray lines

        # Draw all particles
        for p in particles:
            pygame.draw.circle(
                self.screen, 
                p.color, 
                (int(p.x), int(p.y)), 
                int(p.radius)
            )

        # Display stats if available and valid
        if isinstance(stats, dict) and self.font:
            fps = 1.0 / stats["avg_frame_time"] if stats["avg_frame_time"] > 0 else 0
            stats_text = f"FPS: {fps:.1f} | Collisions/frame: {stats['avg_collisions']:.1f} | Particles: {len(particles)} | Threads: {NUM_THREADS}"
            avg_stats_text = f"Avg frame time: {stats['avg_frame_time']*1000:.2f} ms, Avg collisions/frame: {stats['avg_collisions']:.2f}" 

            text_surface = self.font.render(stats_text, True, FONT_COLOR)
            avg_stats_surface = self.font.render(avg_stats_text, True, FONT_COLOR)

            self.screen.blit(text_surface, (10, 10))
            self.screen.blit(avg_stats_surface, (10, 30))

        pygame.display.flip()


def parse_arguments():
    """
    Parses command-line arguments for configuring the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including number of threads and particles.
    """
    parser = argparse.ArgumentParser(description="Particle Simulation")
    parser.add_argument(
        "-t", "--threads", type=int, default=NUM_THREADS,
        help="Number of threads to use for the simulation (default: 8)"
    )
    parser.add_argument(
        "-p", "--particles", type=int, default=PARTICLE_COUNT,
        help="Number of particles in the simulation (default: 1000)"
    )
    parser.add_argument(
        "-b", "--benchmark", action="store_true",
        help="Run the simulation in benchmark mode (no rendering)"
    )
    parser.add_argument(
        "-e", "--extended_benchmark", action="store_true",
        help="Run the extended benchmark with multiple configurations"
    )
    return parser.parse_args()

def run_pygame_simulation():
    """
    Main entry point for the simulation.
    Sets up the simulation components and runs the main loop.
    """
    args = parse_arguments()  # Parse command-line arguments

    global NUM_THREADS, PARTICLE_COUNT, VELOCITY_RANGE, RADIUS_RANGE, MASS_RADIUS_RANGE
    NUM_THREADS = args.threads
    PARTICLE_COUNT = args.particles

    # Adjust parameters based on updated PARTICLE_COUNT
    if PARTICLE_COUNT > 2000:
        print(f"Warning: High particle count ({PARTICLE_COUNT}) may affect performance.")
        VELOCITY_RANGE = (-100, 100)  # Adjust velocity range for larger particle counts
        MASS_RADIUS_RANGE = RADIUS_RANGE = (2, 4)  # Adjust radius range for larger particle counts

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Particle Simulation")
    clock = pygame.time.Clock()

    # Create the modular components
    use_threading = True  # Set to False to use single-threaded version

    if use_threading:
        particle_system = MultithreadedParticleSystem(SCREEN_WIDTH, SCREEN_HEIGHT, num_threads=NUM_THREADS)
        print(f"Using multithreaded system with {NUM_THREADS} threads for {PARTICLE_COUNT} particles")
    else:
        particle_system = ParticleSystem(SCREEN_WIDTH, SCREEN_HEIGHT)
        print("Using single-threaded system for {PARTICLE_COUNT} particles")

    particle_system.create_particles(PARTICLE_COUNT)
    renderer = ParticleRenderer(screen)

    # Main loop variables
    running = True
    paused = False  # Add a paused state
    show_stats = True
    draw_boundaries = False
    stats_update_timer = 0

    while running:
        dt = clock.tick(FRAME_RATE) / 1000.0  # Time in seconds since last frame
        stats_update_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # Exit on ESC
                    running = False 
                elif event.key == pygame.K_SPACE:  # Toggle pause on SPACE
                    paused = not paused
                elif event.key == pygame.K_s:  # Toggle stats display
                    show_stats = not show_stats
                elif event.key == pygame.K_g:  # Toggle boundary display
                    draw_boundaries = not draw_boundaries
                elif event.key == pygame.K_r:  # Reset simulation
                    particle_system.create_particles(PARTICLE_COUNT)

        if not paused:  # Only update simulation if not paused
            particle_system.update(dt)

        if show_stats:
            stats = particle_system.get_profiler_stats()
            renderer.render(particle_system.particles, stats, draw_boundaries)

            if stats_update_timer >= 1.0:
                particle_system.profiler.print_stats()
                stats_update_timer = 0
        else:
            renderer.render(particle_system.particles, draw_boundaries)

    pygame.quit()


def main():
    """
    Main entry point for the simulation.
    Parses arguments and runs either the pygame simulation, the benchmark, or the extended benchmark.
    """
    args = parse_arguments()  # Parse command-line arguments

    global NUM_THREADS, PARTICLE_COUNT
    NUM_THREADS = args.threads
    PARTICLE_COUNT = args.particles

    # Replace the existing benchmarking code with calls to the imported methods
    if hasattr(args, 'benchmark') and args.benchmark:
        run_benchmark_simulation(
            MultithreadedParticleSystem,
            10.0, 
            PARTICLE_COUNT, 
            NUM_THREADS, 
            SCREEN_WIDTH, 
            SCREEN_HEIGHT, 
            FRAME_RATE
            )  # Run benchmark for 10 seconds
    elif hasattr(args, 'extended_benchmark') and args.extended_benchmark:
        run_extended_benchmark(
            MultithreadedParticleSystem,
            SCREEN_WIDTH, 
            SCREEN_HEIGHT, 
            FRAME_RATE
        )  # Run extended benchmark
    else:
        run_pygame_simulation()

if __name__ == "__main__":
    main()