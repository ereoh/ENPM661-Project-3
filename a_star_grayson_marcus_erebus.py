import numpy as np
from collections import deque
import sys
import heapq
import math
import pygame

# Define colors
red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
grey = (128, 128, 128)

# define space resolution
ANGLE_RESOLUTION = 30
STEP_RESOLUTION = 0.5

# Map Width and Height
WIDTH = 600
HEIGHT = 250

# Scaling factor for the animation
SCALE = 3  # Increase the size of the animation by a factor of 2

# Base class for obstacles
class Obstacle:
    def is_inside_obstacle(self, x, y):
        raise NotImplementedError("This method should be overridden by subclasses")

# Class to define line constraints for a convex polygon
class LineDefinedObstacle:
    def __init__(self, vertices):
        self.lines = self.compute_line_constraints(vertices)

    def compute_line_constraints(self, vertices):
        # Compute the line constraints for the rectangle
        lines = []
        num_vertices = len(vertices)
        
        for i in range(num_vertices):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % num_vertices]  # Next vertex in the sequence
            
            # Compute the line equation ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = -(a * x1 + b * y1)
            
            # Determine which side is "inside" by checking the third point
            x_test, y_test = vertices[(i + 2) % num_vertices]
            side = 1 if (a * x_test + b * y_test + c) > 0 else -1
            
            lines.append((a, b, c, side))
        
        return lines
    
    def is_inside_obstacle(self, x, y):
        # Check if the point is on the correct side of all lines
        for a, b, c, side in self.lines:
            value = a * x + b * y + c
            if side == 1 and value < 0:
                return False  # Should be on the left but is on the right
            if side == -1 and value > 0:
                return False  # Should be on the right but is on the left
        return True

# Circular obstacle
class Circle(Obstacle):
    def __init__(self, center, radius):
        # Initialize with center and radius
        self.center = center
        self.radius = radius

    def is_inside_obstacle(self, x, y):
        # Check if point is inside circle
        return (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= self.radius ** 2

# E-shaped obstacle
class EObstacle(Obstacle):
    def __init__(self, origin, width, height):
        # Initialize with origin, width, and height
        self.origin = origin
        self.width = width
        self.height = height
        # Define bars and spine of E shape
        self.bottom_bar = LineDefinedObstacle([
            (origin[0] + 6, origin[1]), 
            (origin[0] + width, origin[1]), 
            (origin[0] + width, origin[1] + 6), 
            (origin[0] + 6, origin[1] + 6)
        ])
        self.top_bar = LineDefinedObstacle([
            (origin[0] + 6, origin[1] + height - 6), 
            (origin[0] + width, origin[1] + height - 6), 
            (origin[0] + width, origin[1] + height), 
            (origin[0] + 6, origin[1] + height)
        ])
        self.middle_bar = LineDefinedObstacle([
            (origin[0] + 6, (origin[1] + height // 2) - 3), 
            (origin[0] + width, (origin[1] + height // 2) - 3), 
            (origin[0] + width, (origin[1] + height // 2) + 3), 
            (origin[0] + 6, (origin[1] + height // 2) + 3)
        ])
        self.spine = LineDefinedObstacle([
            (origin[0], origin[1]), 
            (origin[0] + 6, origin[1]), 
            (origin[0] + 6, origin[1] + height), 
            (origin[0], origin[1] + height)
        ])

    def is_inside_obstacle(self, x, y):
        # Check if point is inside any part of E shape
        return (self.bottom_bar.is_inside_obstacle(x, y) or 
                self.top_bar.is_inside_obstacle(x, y) or 
                self.middle_bar.is_inside_obstacle(x, y) or 
                self.spine.is_inside_obstacle(x, y))

# N-shaped obstacle
class NObstacle(Obstacle): 
    def __init__(self, origin, width, height):
        # Initialize with origin, width, and height
        self.origin = origin
        self.width = width
        self.height = height
        # Define bars and diagonal of N shape
        self.left_bar = LineDefinedObstacle([
            (origin[0], origin[1]), 
            (origin[0] + 6, origin[1]), 
            (origin[0] + 6, origin[1] + height), 
            (origin[0], origin[1] + height)
        ])
        self.right_bar = LineDefinedObstacle([
            (origin[0] + width - 6, origin[1]), 
            (origin[0] + width, origin[1]), 
            (origin[0] + width, origin[1] + height), 
            (origin[0] + width - 6, origin[1] + height)
        ])
        self.diagonal_bar = LineDefinedObstacle([
            (origin[0] + 6, origin[1] + height),
            (origin[0] + 6, origin[1] + height - 20),
            (origin[0] + width - 6, origin[1]),
            (origin[0] + width - 6, origin[1] + 20)
        ])
    
    def is_inside_obstacle(self, x, y):
        # Check if point is inside any part of N shape
        return (self.left_bar.is_inside_obstacle(x, y) or 
                self.right_bar.is_inside_obstacle(x, y) or 
                self.diagonal_bar.is_inside_obstacle(x, y))

# P-shaped obstacle
class PObstacle(Obstacle):
    def __init__(self, origin, width, height):
        # Initialize with origin, width, and height
        self.origin = origin
        self.width = width
        self.height = height
        # Define spine, head, and circle of P shape
        self.spine = LineDefinedObstacle([
            (origin[0], origin[1]), 
            (origin[0] + 6, origin[1]), 
            (origin[0] + 6, origin[1] + height), 
            (origin[0], origin[1] + height)
        ])
        self.head = LineDefinedObstacle([
            (origin[0] + 6, origin[1] + height),
            (origin[0] + 6, origin[1] + height - 50),
            (origin[0] + width - 6, origin[1] + height - 50),
            (origin[0] + width - 6, origin[1] + height)
        ])
        self.circle = Circle((origin[0] + width - 6, origin[1] + height - 25), 25)

    def is_inside_obstacle(self, x, y):
        # Check if point is inside any part of P shape
        return (self.spine.is_inside_obstacle(x, y) or 
                self.head.is_inside_obstacle(x, y) or 
                self.circle.is_inside_obstacle(x, y))

# M-shaped obstacle
class MObstacle(Obstacle):
    def __init__(self, origin, width, height):
        # Initialize with origin, width, and height
        self.origin = origin
        self.width = width
        self.height = height
        # Define bars and diagonals of M shape
        self.left_bar = LineDefinedObstacle([
            (origin[0], origin[1]), 
            (origin[0] + 6, origin[1]), 
            (origin[0] + 6, origin[1] + height), 
            (origin[0], origin[1] + height)
        ])
        self.right_bar = LineDefinedObstacle([
            (origin[0] + width - 6, origin[1]), 
            (origin[0] + width, origin[1]), 
            (origin[0] + width, origin[1] + height), 
            (origin[0] + width - 6, origin[1] + height)
        ])
        self.left_diagonal = LineDefinedObstacle([
            (origin[0] + 6, origin[1] + height),
            (origin[0] + 6, origin[1] + height - 20),
            (origin[0] + width // 2, origin[1] + height // 2),
            (origin[0] + width // 2, origin[1] + height // 2 + 20)
        ])
        self.right_diagonal = LineDefinedObstacle([
            (origin[0] + width - 6, origin[1] + height),
            (origin[0] + width - 6, origin[1] + height - 20),
            (origin[0] + width // 2, origin[1] + height // 2),
            (origin[0] + width // 2, origin[1] + height // 2 + 20)
        ])
    
    def is_inside_obstacle(self, x, y):
        # Check if point is inside any part of M shape
        return (self.left_bar.is_inside_obstacle(x, y) or 
                self.right_bar.is_inside_obstacle(x, y) or 
                self.left_diagonal.is_inside_obstacle(x, y) or 
                self.right_diagonal.is_inside_obstacle(x, y))

# 6-shaped obstacle
class SixObstacle(Obstacle):
    def __init__(self, origin, width, height):
        # Initialize with origin, width, and height
        self.origin = origin
        self.width = width
        self.height = height
        # Define base, spine, and top bar of 6 shape
        self.base = LineDefinedObstacle([
            (origin[0], origin[1]),
            (origin[0] + width, origin[1]),
            (origin[0] + width, origin[1] + height // 2),
            (origin[0], origin[1] + height // 2)
        ])
        self.spine = LineDefinedObstacle([
            (origin[0], origin[1] + height // 2),
            (origin[0] + 6, origin[1] + height // 2),
            (origin[0] + 6, origin[1] + height),
            (origin[0], origin[1] + height)
        ])
        self.top_bar = LineDefinedObstacle([
            (origin[0] + 6, origin[1] + height - 6),
            (origin[0] + width, origin[1] + height - 6),
            (origin[0] + width, origin[1] + height),
            (origin[0] + 6, origin[1] + height)
        ])

    def is_inside_obstacle(self, x, y):
        # Check if point is inside any part of 6 shape
        return (self.top_bar.is_inside_obstacle(x, y) or 
                self.base.is_inside_obstacle(x, y) or 
                self.spine.is_inside_obstacle(x, y))

# 1-shaped obstacle
class OneObstacle(Obstacle):
    def __init__(self, origin, width, height):
        # Initialize with origin, width, and height
        self.origin = origin
        self.width = width
        self.height = height
        # Define bar of 1 shape
        self.bar = LineDefinedObstacle([
            (origin[0], origin[1]),
            (origin[0] + width, origin[1]),
            (origin[0] + width, origin[1] + height),
            (origin[0], origin[1] + height)
        ])
    
    def is_inside_obstacle(self, x, y):
        # Check if point is inside bar of 1 shape
        return self.bar.is_inside_obstacle(x, y)

def discretize(x, y, theta, pos_res=STEP_RESOLUTION, angle_res=ANGLE_RESOLUTION):
    x_d = round(x / pos_res) * pos_res
    y_d = round(y / pos_res) * pos_res
    theta_d = round(theta / angle_res) * angle_res % 360
    return (x_d, y_d, theta_d)

def is_valid_point(x, y, obstacles):
    """
    Check if a point is within canvas bounds and not inside any obstacle.
    """
    # Check canvas bounds
    if not (5 <= x <= (WIDTH - 6) and 5 <= y <= (HEIGHT - 6)):
        return False

    # Check if the point is inside any obstacle
    for obstacle in obstacles:
        if obstacle.is_inside_obstacle(x, y):
            return False

    return True

def astar_search(start, goal, obstacles, L):
    """
    Perform A* search to find the shortest path from start to goal.
    Returns the path, visited matrix, and the order of exploration.
    """
    move_angles = [-60, -30, 0, 30, 60]
    move_cache = {angle: (L * math.cos(math.radians(angle)),
                          L * math.sin(math.radians(angle)))
                  for angle in range(0, 360, ANGLE_RESOLUTION)}

    start_node = discretize(*start)  # Ensure start is discretized
    goal_node = discretize(goal[0], goal[1], 0)  # Discretize goal, ignoring orientation
    goal_pos = goal_node[:2]

    def heuristic(a, b):
        """Calculate Euclidean distance as the heuristic."""
        return math.hypot(b[0] - a[0], b[1] - a[1])

    open_set = []
    heapq.heappush(open_set, (0, start_node))
    C2C = {start_node: 0}
    total_cost = {start_node: heuristic(start_node, goal_pos)}
    parent = {}
    visited = np.zeros((500, 1200, 12), dtype=bool)  # 3D matrix for visited nodes
    exploration_order = []  # Track the order of exploration

    while open_set:
        _, current = heapq.heappop(open_set)
        x, y, theta = current
        key = discretize(x, y, theta)  # Ensure current node is discretized

        # Convert discretized coordinates to matrix indices
        x_idx = int(x * 2)  # Scale x to fit 1200
        y_idx = int(y * 2)  # Scale y to fit 500
        theta_idx = int(theta / ANGLE_RESOLUTION)

        if visited[y_idx, x_idx, theta_idx]:
            continue
        visited[y_idx, x_idx, theta_idx] = True

        # Add to exploration order only if it hasn't been visited before
        exploration_order.append(current)

        # Check if the goal is reached (ignore orientation)
        if heuristic((x, y), goal_pos) < 1.5:
            goal_node = (x, y, theta)  # Use the current node as the goal node
            break

        # Explore neighbors
        for delta_angle in move_angles:
            new_theta = (theta + delta_angle) % 360
            dx, dy = move_cache[new_theta]
            new_x, new_y = x + dx, y + dy
            new_key = discretize(new_x, new_y, new_theta)  # Discretize new node

            # Convert new discretized coordinates to matrix indices
            new_x_idx = int(new_key[0] * 2)
            new_y_idx = int(new_key[1] * 2)
            new_theta_idx = int(new_key[2] / ANGLE_RESOLUTION)

            if (new_x_idx < 0 or new_x_idx >= 1200 or
                new_y_idx < 0 or new_y_idx >= 500 or
                visited[new_y_idx, new_x_idx, new_theta_idx] or
                not is_valid_point(new_key[0], new_key[1], obstacles)):
                continue

            tentative_C2C = C2C[key] + L
            if new_key not in C2C or tentative_C2C < C2C[new_key]:
                C2C[new_key] = tentative_C2C
                total_cost[new_key] = tentative_C2C + heuristic(new_key, goal_pos)
                heapq.heappush(open_set, (total_cost[new_key], new_key))
                parent[new_key] = key

    # Backtrack to construct the path
    path = []
    key = goal_node
    while key in parent:
        path.append(key)
        key = parent[key]
    path.reverse()

    print("Path found!")
    return path, visited, exploration_order

def draw_obstacles(screen, obstacles):
    """
    Draw obstacles on the pygame screen using the is_valid_point function.
    Each pixel is checked to determine if it is part of an obstacle.
    """
    for x in range(WIDTH):
        for y in range(HEIGHT):
            # Flip y-axis for pygame and check if the point is invalid
            if not is_valid_point(x, y, obstacles):  # No need to flip y here
                screen.set_at((int(x * SCALE), int((HEIGHT - y) * SCALE)), grey)  # Flip y for rendering

def animate_astar(screen, obstacles, path, exploration_order, move_cache, L):
    """
    Animate the A* search process, preserving the order of exploration.
    - Visited nodes are marked with blue dots in the order they are explored.
    - Valid moves from each node are drawn as green vectors (lines) at angles -60, -30, 0, 30, and 60 degrees.
    - The final path is drawn in red.
    """
    move_angles = [-60, -30, 0, 30, 60]  # Define the possible move angles

    for node in exploration_order:
        x, y, theta = node
        # Mark visited nodes with blue dots
        pygame.draw.circle(screen, blue, (int(x * SCALE), int((HEIGHT - y) * SCALE)), 2)

        # Draw valid moves as green lines (vectors)
        for delta_angle in move_angles:
            new_theta = (theta + delta_angle) % 360
            dx = L * math.cos(math.radians(new_theta))
            dy = L * math.sin(math.radians(new_theta))
            new_x, new_y = x + dx, y + dy

            if is_valid_point(new_x, new_y, obstacles):
                pygame.draw.line(screen, green, 
                                 (int(x * SCALE), int((HEIGHT - y) * SCALE)), 
                                 (int(new_x * SCALE), int((HEIGHT - new_y) * SCALE)), 2)  # Scale coordinates
        
        pygame.draw.rect(screen, (255, 0, 0), (int(path[-1][0]) * SCALE, (HEIGHT - int(path[-1][1])) * SCALE, 5, 5))
        pygame.draw.rect(screen, (255, 0, 255), (int(path[0][0]) * SCALE, (HEIGHT - int(path[0][1])) * SCALE, 5, 5))

        pygame.display.flip()
        pygame.time.delay(10)  # Delay for animation effect

    # Draw the final path in red
    for i in range(len(path) - 1):
        x1, y1, _ = path[i]
        x2, y2, _ = path[i + 1]
        pygame.draw.line(screen, red, 
                         (int(x1 * SCALE), int((HEIGHT - y1) * SCALE)), 
                         (int(x2 * SCALE), int((HEIGHT - y2) * SCALE)), 3)  # Scale coordinates
    pygame.display.flip()

def visualize_astar(obstacles, path, exploration_order, move_cache, L):
    """
    Visualize the A* search process and the final path using pygame.
    - Initializes the pygame screen.
    - Draws the obstacles on the map.
    - Animates the A* search process and the final path.
    """
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))  # Scale the window size
    pygame.display.set_caption("A* Pathfinding Animation")
    clock = pygame.time.Clock()

    # Fill the screen with a white background
    screen.fill((255, 255, 255))
    # Draw obstacles on the screen
    draw_obstacles(screen, obstacles)
    

    pygame.display.flip()
    
    

    # Animate the A* search process
    animate_astar(screen, obstacles, path, exploration_order, move_cache, L)

    
    
    # Keep the visualization open until the user closes it
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(30)

    pygame.quit()

# Constants for obstacle dimensions
E_WIDTH, E_HEIGHT = 50, 150
N_WIDTH, N_HEIGHT = 50, 150
P_WIDTH, P_HEIGHT = 50, 150
M_WIDTH, M_HEIGHT = 50, 150
SIX_WIDTH, SIX_HEIGHT = 50, 150
ONE_WIDTH, ONE_HEIGHT = 30, 150

# Initialize obstacles
E_obstacle = EObstacle((80, 50), E_WIDTH, E_HEIGHT)
N_obstacle = NObstacle((150, 50), N_WIDTH, N_HEIGHT)
P_Obstacle = PObstacle((220, 50), P_WIDTH, P_HEIGHT)
M_Obstacle = MObstacle((300, 50), M_WIDTH, M_HEIGHT)
Six_Obstacle_1 = SixObstacle((370, 50), SIX_WIDTH, SIX_HEIGHT)
Six_Obstacle_2 = SixObstacle((440, 50), SIX_WIDTH, SIX_HEIGHT)
One_Obstacle = OneObstacle((510, 50), ONE_WIDTH, ONE_HEIGHT)

Obstacles = [E_obstacle, N_obstacle, P_Obstacle, M_Obstacle, Six_Obstacle_1, Six_Obstacle_2, One_Obstacle]

# Request start and goal points from the command line
try:
    print("Maze: Bottom left corner is (5, 5) and top right corner is (", (WIDTH - 6), ",", (HEIGHT - 6), ")")

    valid = False
    while not valid:
        start_x, start_y = map(int, input("Enter start point (x y): ").split())
        if is_valid_point(start_x, start_y, Obstacles):
            valid = True
        else:
            print("Invalid start point. Ensure it is within bounds and not inside an obstacle.")

    valid = False
    while not valid:
        goal_x, goal_y = map(int, input("Enter goal point (x y): ").split())
        if is_valid_point(goal_x, goal_y, Obstacles):
            valid = True
        else:
            print("Invalid goal point. Ensure it is within bounds and not inside an obstacle.")
except ValueError:
    print("Invalid input. Please enter integer coordinates.")
    sys.exit(1)

start = (start_x, start_y, 0)
goal = (goal_x, goal_y, 0)

# Perform A* search and get the path
path, closed_set, exploration_order = astar_search(start, goal, Obstacles, 10) # Changed step from 5 to 10
print(len(exploration_order))
print(path[0])

# Visualize the A* search process and the final path
move_cache = {angle: (math.cos(math.radians(angle)), math.sin(math.radians(angle))) 
              for angle in range(0, 360, ANGLE_RESOLUTION)}
visualize_astar(Obstacles, path, exploration_order, move_cache, 7)

# Convert to numpy arrays for animation
np_path = np.array(path)
np_closed_set = np.array(list(closed_set), dtype=float)

num_visited = np_closed_set.shape[0]
path_length = len(path)
num_frames = num_visited + path_length
