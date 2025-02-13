import numpy as np
import heapq
from math import exp, sqrt
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import cycle

# Parameters
sigma_0 = 100  # Initial maximum life strength
sigma_threshold = 10  # Minimum life strength for rescue
decay_rate = 0.0037  # Life strength decay rate
robot_speed = 1.0  # Robot speed
rescue_time = 5  # Time required to rescue one survivor

# Define Euclidean distance function
def euclidean_distance(node, goal):
    return sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)
map_size = 50
disaster_center = (map_size//2, map_size//2)  # Disaster center location

L = sqrt(map_size**2 + map_size**2)  # Maximum distance on the map

# Define map size and obstacles
obstacle_map = np.zeros((map_size, map_size))
rectangles = []

# Add random rectangular obstacles
max_obstacles_area = int(map_size * map_size * 0.4)  # Maximum total obstacle area (40% of map)
current_obstacle_area = 0

while current_obstacle_area < max_obstacles_area:
    rect_width = random.randint(1, 10)
    rect_height = random.randint(1, 10)
    start_x = random.randint(0, map_size - rect_width - 1)
    start_y = random.randint(0, map_size - rect_height - 1)

    # Check if this rectangle overlaps with existing ones
    overlap = False
    for (x, y, w, h) in rectangles:
        if not (start_x + rect_width <= x or x + w <= start_x or start_y + rect_height <= y or y + h <= start_y):
            overlap = True
            break

    if not overlap:
        obstacle_map[start_x:start_x + rect_width, start_y:start_y + rect_height] = 1
        rectangles.append((start_x, start_y, rect_width, rect_height))
        current_obstacle_area += rect_width * rect_height

# Generate survivor locations ensuring no overlap with obstacles or each other
survivor_locations = []
while len(survivor_locations) < 10:
    candidate = (random.randint(0, map_size - 1), random.randint(0, map_size - 1))
    if obstacle_map[candidate[0], candidate[1]] == 0 and candidate not in survivor_locations:
        survivor_locations.append(candidate)

num_survivors = len(survivor_locations)
survivor_labels = [chr(65 + i) for i in range(num_survivors)]  # Assign letters A-Z to survivors

# Calculate initial life strengths based on disaster center and formula
initial_life_strengths = []
for location in survivor_locations:
    distance_to_center = euclidean_distance(location, disaster_center)
    life_strength = sigma_0 * min(distance_to_center / L, 1)
    initial_life_strengths.append(life_strength)

# Calculate life strength at a given time
def life_strength(sigma_0, time, decay_rate):
    return sigma_0 * exp(-decay_rate * time)

# A* Algorithm for shortest path
def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current):
            tentative_g_score = g_score[current] + diagonal_cost(current, neighbor)

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + euclidean_distance(neighbor, goal)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

# Get neighbors while avoiding obstacles
def get_neighbors(node):
    neighbors = []
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal moves
    ]
    for d in directions:
        neighbor = (node[0] + d[0], node[1] + d[1])
        if 0 <= neighbor[0] < map_size and 0 <= neighbor[1] < map_size:
            if obstacle_map[neighbor[0], neighbor[1]] == 0:  # Check for obstacle
                if abs(d[0]) == 1 and abs(d[1]) == 1:  # Check diagonal moves
                    # Ensure both adjacent cells are obstacle-free for diagonal move
                    intermediate_1 = (node[0], neighbor[1])
                    intermediate_2 = (neighbor[0], node[1])
                    if (obstacle_map[intermediate_1[0], intermediate_1[1]] == 0 and
                            obstacle_map[intermediate_2[0], intermediate_2[1]] == 0):
                        neighbors.append(neighbor)
                else:
                    neighbors.append(neighbor)
    return neighbors

# Reconstruct path
def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

# Diagonal cost
def diagonal_cost(node, neighbor):
    return sqrt(2) if abs(node[0] - neighbor[0]) == 1 and abs(node[1] - neighbor[1]) == 1 else 1

# Improved Iterated Greedy (IIG) Algorithm
def iig_algorithm(survivor_locations, max_iterations=100):
    current_solution = random.sample(survivor_locations, len(survivor_locations))
    best_solution = current_solution[:]
    best_rescued, rescued_life_strengths = evaluate_solution(best_solution)

    for iteration in range(max_iterations):
        destroyed_solution = destruction_phase(current_solution)
        reconstructed_solution = construction_phase(destroyed_solution, survivor_locations)
        optimized_solution = local_search(reconstructed_solution)
        rescued, life_strengths = evaluate_solution(optimized_solution)

        if rescued > best_rescued:
            best_solution = optimized_solution[:]
            best_rescued = rescued
            rescued_life_strengths = life_strengths

        current_solution = optimized_solution[:]

    return best_solution, best_rescued, rescued_life_strengths

# Destruction Phase
def destruction_phase(solution):
    num_to_remove = len(solution) // 3
    removed = random.sample(solution, num_to_remove)
    return [s for s in solution if s not in removed]

# Construction Phase
def construction_phase(destroyed_solution, survivor_locations):
    remaining = [s for s in survivor_locations if s not in destroyed_solution]
    return destroyed_solution + sorted(remaining, key=lambda s: initial_life_strengths[survivor_locations.index(s)], reverse=True)

# Local Search
def local_search(solution):
    for i in range(len(solution) - 1):
        for j in range(i + 1, len(solution)):
            new_solution = solution[:]
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            if evaluate_solution(new_solution)[0] > evaluate_solution(solution)[0]:
                solution = new_solution[:]
    return solution

# Evaluate Solution
def evaluate_solution(solution):
    total_rescued = 0
    time_elapsed = 0
    rescued_life_strengths = []
    for i in range(len(solution)):
        if i > 0:
            path = a_star(solution[i - 1], solution[i])
            if not path:
                continue
            time_elapsed += len(path) - 1
        life_strength_at_time = life_strength(
            initial_life_strengths[survivor_locations.index(solution[i])],
            time_elapsed,
            decay_rate
        )
        if life_strength_at_time > sigma_threshold:
            total_rescued += 1
            rescued_life_strengths.append((solution[i], life_strength_at_time))
            time_elapsed += rescue_time
    return total_rescued, rescued_life_strengths

# Run the IIG Algorithm
best_solution, best_rescued, rescued_life_strengths = iig_algorithm(survivor_locations)

rescue_labels = [survivor_labels[survivor_locations.index(s)] for s, _ in rescued_life_strengths]
# Return to the first survivor's location
if len(best_solution) > 1:
    return_path = a_star(best_solution[-1], best_solution[0])
    if return_path:
        best_solution.append(best_solution[0])
        rescue_labels.append(rescue_labels[0])


# Print results
print("Survivor Locations:", survivor_locations)
print("Initial Life Strengths:", initial_life_strengths)
print("Best Solution (Rescue Order):", best_solution)
print("Rescue Order (Labels):", rescue_labels)
rescued_life_strengths_with_labels = [(rescue_labels[i], strength) for i, (_, strength) in enumerate(rescued_life_strengths)]
print("Rescued Life Strengths (Labels and Remaining Life):", rescued_life_strengths_with_labels)
print("Number of Rescued Survivors:", best_rescued)

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))

# Plot obstacles as rectangles
for (x, y, w, h) in rectangles:
    rect = Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='gray')
    ax.add_patch(rect)

# Plot survivors with labels
for i, (x, y) in enumerate(survivor_locations):
    ax.scatter(x, y, color="blue", s=100)
    ax.text(x + 0.5, y + 0.5, survivor_labels[i], fontsize=10, color="black")

# Plot disaster center
ax.scatter(disaster_center[0], disaster_center[1], color="red", s=200, label="Disaster Center")

# Plot the path pixel by pixel with different colors
colors = cycle(['green', 'red', 'blue', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow', 'pink'])
for i in range(len(best_solution) - 1):
    path = a_star(best_solution[i], best_solution[i + 1])
    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color=next(colors), linewidth=2)

# Annotate rescued life strengths
for (location, life_strength) in rescued_life_strengths:
    x, y = location
    ax.text(x + 0.5, y - 0.5, f"{life_strength:.1f}", fontsize=8, color="red")

# Configure the plot
ax.set_xlim(0, map_size)
ax.set_ylim(0, map_size)
ax.set_title("Rescue Path with Rectangular Obstacles")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.legend()

plt.show()
