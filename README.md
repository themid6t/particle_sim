# Particle Simulation

## Overview
This project is a particle simulation system implemented in Python. It simulates the movement and collision of particles within a confined space. The simulation supports both single-threaded and multithreaded execution. Visualization is provided using Pygame.

## Installation and Setup

1. **Install Python**: Ensure you have Python 3.8 or later installed.
2. **Install Dependencies**:
   ```bash
   pip install pygame
   ```
3. **Run the Simulation**:
   ```bash
   python physics_simulation.py
   ```

## Usage

1. **Run the Simulation with Default Settings**:
   ```bash
   python physics_simulation.py
   ```

2. **Run the Simulation with Custom Particle Count**:
   ```bash
   python physics_simulation.py -p 2000
   ```
   This runs the simulation with 2000 particles.

3. **Run the Simulation with Custom Number of Threads**:
   ```bash
   python physics_simulation.py -t 4
   ```
   This runs the simulation using 4 threads.

4. **Run the Simulation with Custom Particle Count and Threads**:
   ```bash
   python physics_simulation.py -p 1500 -t 6
   ```
   This runs the simulation with 1500 particles and 6 threads.

## Code Structure

- **`Particle`**: Represents individual particles with properties like position, velocity, radius, mass, and color.

- **`ParticleSystem`**: Handles particle creation, updates, and collision detection using a sweep-and-prune algorithm.

- **`MultithreadedParticleSystem`**: Extends `ParticleSystem` to use multiple threads for simulation. Particles are partitioned by their x-coordinates, and each partition is processed in a separate thread.

- **`Profiler`**: Measures frame times and collision counts for performance analysis.

- **`ParticleRenderer`**: Uses Pygame to render particles and display simulation statistics. Now supports visualizing thread-separated boundaries and buffer regions.

- **`run_pygame_simulation`**: The main entry point for the simulation, setting up the environment and running the main loop.


## Class and Function Documentation

### `Particle`
- **Attributes**:
  - `x`, `y`: Position of the particle.
  - `vx`, `vy`: Velocity of the particle.
  - `radius`: Radius of the particle.
  - `mass`: Mass of the particle, calculated based on its radius.
  - `color`: Color of the particle, determined by its mass.

### `ParticleSystem`
- **Methods**:
  - `create_particles(n)`: Creates `n` particles with random properties.
  - `update(dt)`: Updates particle positions, handles collisions, and returns the updated list of particles.
  - `_handle_particle_collision(p1, p2)`: Resolves collisions between two particles.
  - `_sweep_and_prune()`: Efficiently detects and resolves collisions using the sweep-and-prune algorithm.

### `MultithreadedParticleSystem`
- **Methods**:
  - `update(dt)`: Updates particles using multiple threads.
  - `_partition_particles_with_boundaries(particles, num_parts)`: Partitions particles by x-coordinates and identifies boundary particles.
  - `_handle_boundary_collisions(updated_partitions, boundary_map)`: Handles collisions between particles at partition boundaries. This method uses locks to ensure thread-safe access when processing particles in neighboring partitions, preventing race conditions during collision detection and resolution.

### `Profiler`
- **Description**: A custom profiler was implemented because using `cProfile` significantly reduced simulation performance.
- **Methods**:
  - `start_frame()`: Starts timing a frame.
  - `end_frame()`: Ends timing a frame and records statistics.
  - `record_collision()`: Records a collision event.
  - `get_stats()`: Returns average frame time and collision statistics.

### `ParticleRenderer`
- **Methods**:
  - `render(particles, stats)`: Renders particles and displays statistics on the screen.

### `run_pygame_simulation`
- Initializes the simulation and runs the main loop.
- Supports toggling statistics display (`S` key) and (`G` key) to toggle thread spaces and resetting the simulation (`R` key).

## Performance Considerations
- The simulation uses multithreading to improve performance for large particle counts.
- **However, due to Python's Global Interpreter Lock (GIL), threads are executed concurrently rather than in true parallel, which limits the performance gains from multithreading.**
- The number of threads can be adjusted using the `NUM_THREADS` constant.
- For very high particle counts, the velocity and radius ranges are automatically adjusted to maintain performance.

## Future Improvements
- Add multiprocessing instead, to get parallel processing.
