import time
import json
import subprocess
import matplotlib.pyplot as plt
import os

EXTENDED_BENCHMARK_CONFIGURATIONS = [
    {"particles": 1000, "threads": 1},
    {"particles": 1000, "threads": 4},
    {"particles": 1000, "threads": 8},
    {"particles": 2500, "threads": 1}, 
    {"particles": 2500, "threads": 4}, 
    {"particles": 2500, "threads": 8}, 
    {"particles": 5000, "threads": 1}, 
    {"particles": 5000, "threads": 4}, 
    {"particles": 5000, "threads": 8}, 
    {"particles": 7500, "threads": 1}, 
    {"particles": 7500, "threads": 4}, 
    {"particles": 7500, "threads": 8}, 
    {"particles": 10000, "threads": 1}, 
    {"particles": 10000, "threads": 4}, 
    {"particles": 10000, "threads": 8},
]

OUTPUT_FOLDER = "analysis_results"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def run_benchmark_simulation(engine_class, duration: float, particle_count: int, num_threads: int, screen_width: int, screen_height: int, frame_rate: int):
    """
    Runs the simulation in benchmark mode for a specified duration.

    Args:
        engine_class: The particle system engine class to use.
        duration (float): Duration of the benchmark in seconds.
        particle_count (int): Number of particles in the simulation.
        num_threads (int): Number of threads to use for the simulation.
        screen_width (int): Width of the simulation area.
        screen_height (int): Height of the simulation area.
        frame_rate (int): Frame rate of the simulation.
    """
    particle_system = engine_class(screen_width, screen_height, num_threads=num_threads)
    print(f"Benchmarking with {num_threads} threads for {particle_count} particles")

    particle_system.create_particles(particle_count)

    elapsed_time = 0
    frame_count = 0
    start_time = time.time()

    while elapsed_time < duration:
        dt = 1.0 / frame_rate
        particle_system.update(dt)
        frame_count += 1
        elapsed_time = time.time() - start_time

    stats = particle_system.get_profiler_stats()
    print(f"Benchmark Results:")
    print(f"  Total Frames: {frame_count}")
    print(f"  Avg Frame Time: {stats['avg_frame_time']*1000:.2f} ms")
    print(f"  Avg Collisions/Frame: {stats['avg_collisions']:.2f}")
    print(f"  Total Duration: {elapsed_time:.2f} seconds")

def run_extended_benchmark(engine_class, screen_width: int, screen_height: int, frame_rate: int):
    """
    Runs the simulation in benchmark mode for multiple configurations of particles and threads.
    Saves the results to a JSON file for analysis.

    Args:
        engine_class: The particle system engine class to use.
        screen_width (int): Width of the simulation area.
        screen_height (int): Height of the simulation area.
        frame_rate (int): Frame rate of the simulation.
    """
    results = []

    for config in EXTENDED_BENCHMARK_CONFIGURATIONS:
        particle_count = config["particles"]
        num_threads = config["threads"]

        particle_system = engine_class(screen_width, screen_height, num_threads=num_threads)
        print(f"Benchmarking with {num_threads} threads for {particle_count} particles")
        particle_system.create_particles(particle_count)

        elapsed_time = 0
        frame_count = 0
        duration = 10.0
        start_time = time.time()

        while elapsed_time < duration:
            dt = 1.0 / frame_rate
            particle_system.update(dt)
            frame_count += 1
            elapsed_time = time.time() - start_time

        stats = particle_system.get_profiler_stats()
        res = {
            "particles": particle_count,
            "threads": num_threads,
            "total_frames": frame_count,
            "avg_frame_time_ms": stats["avg_frame_time"] * 1000,
            "avg_collisions_per_frame": stats["avg_collisions"],
            "total_duration_s": elapsed_time,
        }
        results.append(res)
        print(json.dumps(res, indent=4))

    with open(f"{OUTPUT_FOLDER}/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Benchmark completed. Results saved to 'benchmark_results.json'.")

    # Call the analysis function after the extended benchmark
    analyze_benchmark(results)


def plot_frame_time_vs_particles(results):
    """Plot average frame time vs. particle count for each thread count."""
    thread_groups = {}
    for result in results:
        threads = result["threads"]
        if threads not in thread_groups:
            thread_groups[threads] = []
        thread_groups[threads].append(result)

    plt.figure()
    for threads, data in thread_groups.items():
        data = sorted(data, key=lambda x: x["particles"])
        particle_counts = [d["particles"] for d in data]
        frame_times = [d["avg_frame_time_ms"] for d in data]
        plt.plot(particle_counts, frame_times, label=f"{threads} Threads")

    plt.xlabel("Particle Count")
    plt.ylabel("Average Frame Time (ms)")
    plt.title("Frame Time vs. Particle Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_FOLDER, "frame_time_vs_particles.png"))
    plt.close()

def plot_frame_time_vs_threads(results):
    """Plot average frame time vs. thread count for each particle count."""
    particle_groups = {}
    for result in results:
        particles = result["particles"]
        if particles not in particle_groups:
            particle_groups[particles] = []
        particle_groups[particles].append(result)

    plt.figure()
    for particles, data in particle_groups.items():
        data = sorted(data, key=lambda x: x["threads"])
        thread_counts = [d["threads"] for d in data]
        frame_times = [d["avg_frame_time_ms"] for d in data]
        plt.plot(thread_counts, frame_times, label=f"{particles} Particles")

    plt.xlabel("Thread Count")
    plt.ylabel("Average Frame Time (ms)")
    plt.title("Frame Time vs. Thread Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_FOLDER, "frame_time_vs_threads.png"))
    plt.close()

def plot_collisions_vs_particles(results):
    """Plot average collisions per frame vs. particle count for each thread count."""
    thread_groups = {}
    for result in results:
        threads = result["threads"]
        if threads not in thread_groups:
            thread_groups[threads] = []
        thread_groups[threads].append(result)

    plt.figure()
    for threads, data in thread_groups.items():
        data = sorted(data, key=lambda x: x["particles"])
        particle_counts = [d["particles"] for d in data]
        collisions = [d["avg_collisions_per_frame"] for d in data]
        plt.plot(particle_counts, collisions, label=f"{threads} Threads")

    plt.xlabel("Particle Count")
    plt.ylabel("Average Collisions per Frame")
    plt.title("Collisions vs. Particle Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_FOLDER, "collisions_vs_particles.png"))
    plt.close()

def analyze_benchmark(results):
    """Perform analysis and save plots."""
    plot_frame_time_vs_particles(results)
    plot_frame_time_vs_threads(results)
    plot_collisions_vs_particles(results)
    print("Analysis complete. Plots saved as PNG files.")