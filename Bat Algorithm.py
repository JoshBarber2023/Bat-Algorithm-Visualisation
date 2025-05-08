import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon, Wedge
import os
import imageio.v2 as imageio

# --- CONFIGURATION ---
NUM_BATS = 10
MAX_ITER = 30
STEPS_PER_ITER = 10
BOUNDS = [-3, 3]
FREQ_MIN, FREQ_MAX = 0, 2
ALPHA = 0.9
GAMMA = 0.9

# --- OBJECTIVE FUNCTION ---
def objective(x, y):
    return x**2 + y**2

# --- BAT CLASS ---
class Bat:
    def __init__(self):
        self.position = np.random.uniform(BOUNDS[0], BOUNDS[1], 2)
        self.velocity = np.zeros(2)
        self.freq = 0
        self.loudness = 1
        self.pulse_rate = np.random.uniform(0, 1)
        self.fitness = objective(*self.position)
        self.next_position = self.position.copy()
        self.pulsing = False

    def update_target(self, global_best, avg_loudness):
        self.freq = FREQ_MIN + (FREQ_MAX - FREQ_MIN) * np.random.rand()
        self.velocity += (self.position - global_best.position) * self.freq
        proposed_position = self.position + self.velocity

        self.pulsing = False
        pulse_direction = None
        if np.random.rand() > self.pulse_rate:
            epsilon = np.random.uniform(-1, 1, 2)
            proposed_position = global_best.position + epsilon * avg_loudness
            self.pulsing = True
            pulse_direction = epsilon / np.linalg.norm(epsilon)

        proposed_position = np.clip(proposed_position, BOUNDS[0], BOUNDS[1])
        new_fitness = objective(*proposed_position)

        if new_fitness < self.fitness and np.random.rand() < self.loudness:
            self.next_position = proposed_position
            self.fitness = new_fitness
            self.loudness *= ALPHA
            self.pulse_rate *= (1 - np.exp(-GAMMA))
        else:
            self.next_position = self.position.copy()

        return pulse_direction

    def step(self, t):
        self.position = (1 - t) * self.position + t * self.next_position

# --- INITIALIZE BATS ---
bats = [Bat() for _ in range(NUM_BATS)]
global_best = min(bats, key=lambda b: b.fitness)

# --- PLOTTING SETUP ---
fig, (ax, ax_fitness) = plt.subplots(1, 2, figsize=(14, 6))
X, Y = np.meshgrid(np.linspace(BOUNDS[0], BOUNDS[1], 200),
                   np.linspace(BOUNDS[0], BOUNDS[1], 200))
Z = objective(X, Y)
ax.contourf(X, Y, Z, levels=50, cmap='viridis')

bat_icons = []
sonar_cones = []
pulse_directions = [np.array([1.0, 0.0])] * NUM_BATS

for _ in range(NUM_BATS):
    triangle = RegularPolygon((0, 0), numVertices=3, radius=0.15, orientation=np.pi / 2,
                              color='red', ec='black')
    ax.add_patch(triangle)
    bat_icons.append(triangle)

    cone = Wedge(center=(0, 0), r=0.1, theta1=0, theta2=30, color='cyan', alpha=0.0)
    ax.add_patch(cone)
    sonar_cones.append(cone)

best_dot, = ax.plot([], [], 'y*', markersize=15, label='Global Best')
ax.set_xlim(BOUNDS)
ax.set_ylim(BOUNDS)
ax.set_title("Bat Algorithm - Animated with Sonar Cones")
ax.legend()

# Fitness tracking
best_fitness_over_time = []

# --- Create directory for saving frames ---
frames_dir = "bat_frames"
os.makedirs(frames_dir, exist_ok=True)

frame_count = 0
ring_lifetimes = [0] * NUM_BATS

# --- ANIMATION FUNCTION ---
def animate(_):
    global frame_count, global_best

    if frame_count >= MAX_ITER * STEPS_PER_ITER:  # Stop animation after max frames
        # Create the GIF once the limit is reached
        create_gif_from_frames()
        ani.event_source.stop()
        return []

    t = (frame_count % STEPS_PER_ITER) / STEPS_PER_ITER

    if frame_count % STEPS_PER_ITER == 0:
        avg_loudness = np.mean([b.loudness for b in bats])
        for i, b in enumerate(bats):
            pulse_dir = b.update_target(global_best, avg_loudness)
            if pulse_dir is not None:
                pulse_directions[i] = pulse_dir
                ring_lifetimes[i] = STEPS_PER_ITER

        new_global = min(bats, key=lambda b: b.fitness)
        if new_global.fitness < global_best.fitness:
            global_best = new_global

        # Track best fitness for plotting
        best_fitness_over_time.append(global_best.fitness)

    for i, b in enumerate(bats):
        b.step(t)
        x, y = b.position
        bat_icons[i].xy = (x, y)
        sonar_cones[i].center = (x, y)

        if ring_lifetimes[i] > 0:
            r = 0.1 + 0.6 * (1 - ring_lifetimes[i] / STEPS_PER_ITER)
            alpha = ring_lifetimes[i] / STEPS_PER_ITER
            direction = pulse_directions[i]
            angle = np.degrees(np.arctan2(direction[1], direction[0]))
            sonar_cones[i].set_radius(r)
            sonar_cones[i].theta1 = angle - 20
            sonar_cones[i].theta2 = angle + 20
            sonar_cones[i].set_alpha(alpha)
            ring_lifetimes[i] -= 1
        else:
            sonar_cones[i].set_alpha(0)

    best_dot.set_data(global_best.position[0], global_best.position[1])
    frame_count += 1

    # Save frame as PNG
    frame_filename = os.path.join(frames_dir, f"frame_{frame_count:03d}.png")
    plt.savefig(frame_filename)

    # Update the fitness plot
    ax_fitness.clear()
    ax_fitness.set_title("Best Fitness Over Time", fontsize=14)
    ax_fitness.set_xlabel("Iteration")
    ax_fitness.set_ylabel("Best Fitness")
    ax_fitness.grid(True)
    ax_fitness.plot(best_fitness_over_time, lw=2, c='purple')
    ax_fitness.set_xlim(0, MAX_ITER)
    ax_fitness.set_ylim(0.02, 0)

    return bat_icons + sonar_cones + [best_dot]

# --- Create GIF function ---
def create_gif_from_frames():
    all_frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    images = [imageio.imread(frame) for frame in all_frames]
    imageio.mimsave("bat_animation.gif", images, duration=0.2)

    # Clean up frames directory after GIF is created
    for f in all_frames:
        os.remove(f)
    os.rmdir(frames_dir)

    print("GIF saved as 'bat_animation.gif'")

# --- RUN ANIMATION ---
ani = animation.FuncAnimation(fig, animate, frames=MAX_ITER * STEPS_PER_ITER,
                              interval=50, blit=True)

plt.show()
