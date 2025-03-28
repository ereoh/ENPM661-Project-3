import numpy as np
from collections import deque
import sys
import heapq
import math
import pygame

# Define colors
red, blue, green, grey = (0, 0, 255), (255, 0, 0), (0, 255, 0), (128, 128, 128)

# Resolution and map dimensions
ANGLE_RESOLUTION, STEP_RESOLUTION = 30, 0.5
WIDTH, HEIGHT, SCALE = 600, 250, 3

class Obstacle:
    def is_inside_obstacle(self, x, y):
        raise NotImplementedError

class LineDefinedObstacle:
    def __init__(self, vertices):
        self.lines = self.compute_line_constraints(vertices)

    def compute_line_constraints(self, vertices):
        lines = []
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            a, b, c = y2 - y1, x1 - x2, -(y2 - y1) * x1 - (x1 - x2) * y1
            side = 1 if (a * vertices[(i + 2) % len(vertices)][0] + b * vertices[(i + 2) % len(vertices)][1] + c) > 0 else -1
            lines.append((a, b, c, side))
        return lines

    def is_inside_obstacle(self, x, y):
        return all((a * x + b * y + c) * side >= 0 for a, b, c, side in self.lines)

class Circle(Obstacle):
    def __init__(self, center, radius):
        self.center, self.radius = center, radius

    def is_inside_obstacle(self, x, y):
        return (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= self.radius ** 2

class EObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
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
        return (self.bottom_bar.is_inside_obstacle(x, y) or 
                self.top_bar.is_inside_obstacle(x, y) or 
                self.middle_bar.is_inside_obstacle(x, y) or 
                self.spine.is_inside_obstacle(x, y))

class NObstacle(Obstacle): 
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
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
        return (self.left_bar.is_inside_obstacle(x, y) or 
                self.right_bar.is_inside_obstacle(x, y) or 
                self.diagonal_bar.is_inside_obstacle(x, y))

class PObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
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
        return (self.spine.is_inside_obstacle(x, y) or 
                self.head.is_inside_obstacle(x, y) or 
                self.circle.is_inside_obstacle(x, y))

class MObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
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
        return (self.left_bar.is_inside_obstacle(x, y) or 
                self.right_bar.is_inside_obstacle(x, y) or 
                self.left_diagonal.is_inside_obstacle(x, y) or 
                self.right_diagonal.is_inside_obstacle(x, y))

class SixObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
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
        return (self.top_bar.is_inside_obstacle(x, y) or 
                self.base.is_inside_obstacle(x, y) or 
                self.spine.is_inside_obstacle(x, y))

class OneObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
        self.bar = LineDefinedObstacle([
            (origin[0], origin[1]),
            (origin[0] + width, origin[1]),
            (origin[0] + width, origin[1] + height),
            (origin[0], origin[1] + height)
        ])
    
    def is_inside_obstacle(self, x, y):
        return self.bar.is_inside_obstacle(x, y)

def discretize(x, y, theta, pos_res=STEP_RESOLUTION, angle_res=ANGLE_RESOLUTION):
    return round(x / pos_res) * pos_res, round(y / pos_res) * pos_res, round(theta / angle_res) * angle_res % 360

def is_valid_point(x, y, obstacles):
    if not (5 <= x <= WIDTH - 6 and 5 <= y <= HEIGHT - 6):
        return False
    return not any(obstacle.is_inside_obstacle(x, y) for obstacle in obstacles)

def astar_search(start, goal, obstacles, L):
    move_angles = [-60, -30, 0, 30, 60]
    move_cache = {angle: (L * math.cos(math.radians(angle)), L * math.sin(math.radians(angle))) for angle in range(0, 360, ANGLE_RESOLUTION)}
    start_node, goal_node = discretize(*start), discretize(goal[0], goal[1], 0)
    goal_pos = goal_node[:2]

    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    open_set, C2C, total_cost, parent = [], {start_node: 0}, {start_node: heuristic(start_node, goal_pos)}, {}
    visited = np.zeros((500, 1200, 12), dtype=bool)
    exploration_order = []

    heapq.heappush(open_set, (0, start_node))
    while open_set:
        _, current = heapq.heappop(open_set)
        x, y, theta = current
        key = discretize(x, y, theta)
        x_idx, y_idx, theta_idx = int(x * 2), int(y * 2), int(theta / ANGLE_RESOLUTION)

        if visited[y_idx, x_idx, theta_idx]:
            continue
        visited[y_idx, x_idx, theta_idx] = True
        exploration_order.append(current)

        if heuristic((x, y), goal_pos) < 1.5:
            goal_node = (x, y, theta)
            break

        for delta_angle in move_angles:
            new_theta = (theta + delta_angle) % 360
            dx, dy = move_cache[new_theta]
            new_x, new_y = x + dx, y + dy
            new_key = discretize(new_x, new_y, new_theta)
            new_x_idx, new_y_idx, new_theta_idx = int(new_key[0] * 2), int(new_key[1] * 2), int(new_key[2] / ANGLE_RESOLUTION)

            if (new_x_idx < 0 or new_x_idx >= 1200 or new_y_idx < 0 or new_y_idx >= 500 or
                visited[new_y_idx, new_x_idx, new_theta_idx] or not is_valid_point(new_key[0], new_key[1], obstacles)):
                continue

            tentative_C2C = C2C[key] + L
            if new_key not in C2C or tentative_C2C < C2C[new_key]:
                C2C[new_key], total_cost[new_key] = tentative_C2C, tentative_C2C + heuristic(new_key, goal_pos)
                heapq.heappush(open_set, (total_cost[new_key], new_key))
                parent[new_key] = key

    path, key = [], goal_node
    while key in parent:
        path.append(key)
        key = parent[key]
    path.append(start)  # Ensure the start node is included
    path.reverse()
    return path, visited, exploration_order

def draw_obstacles(screen, obstacles):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if not is_valid_point(x, y, obstacles):
                screen.set_at((int(x * SCALE), int((HEIGHT - y) * SCALE)), grey)

def animate_astar(screen, obstacles, path, exploration_order, move_cache, L):
    move_angles = [-60, -30, 0, 30, 60]
    for node in exploration_order:
        x, y, theta = node
        pygame.draw.circle(screen, blue, (int(x * SCALE), int((HEIGHT - y) * SCALE)), 2)
        for delta_angle in move_angles:
            new_theta = (theta + delta_angle) % 360
            dx = L * math.cos(math.radians(new_theta))
            dy = L * math.sin(math.radians(new_theta))
            new_x, new_y = x + dx, y + dy
            if is_valid_point(new_x, new_y, obstacles):
                pygame.draw.line(screen, green, 
                                 (int(x * SCALE), int((HEIGHT - y) * SCALE)), 
                                 (int(new_x * SCALE), int((HEIGHT - new_y) * SCALE)), 2)
        
        pygame.draw.rect(screen, (255, 0, 0), (int(path[-1][0]) * SCALE, (HEIGHT - int(path[-1][1])) * SCALE, 5, 5))
        pygame.draw.rect(screen, (255, 0, 255), (int(path[0][0]) * SCALE, (HEIGHT - int(path[0][1])) * SCALE, 5, 5))

        pygame.display.flip()
        pygame.time.delay(1)
    for i in range(len(path) - 1):
        x1, y1, _ = path[i]
        x2, y2, _ = path[i + 1]
        pygame.draw.line(screen, red, 
                         (int(x1 * SCALE), int((HEIGHT - y1) * SCALE)), 
                         (int(x2 * SCALE), int((HEIGHT - y2) * SCALE)), 3)
    pygame.display.flip()

def visualize_astar(obstacles, path, exploration_order, move_cache, L):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
    pygame.display.set_caption("A* Pathfinding Animation")
    clock = pygame.time.Clock()
    screen.fill((255, 255, 255))
    draw_obstacles(screen, obstacles)
    

    pygame.display.flip()
    animate_astar(screen, obstacles, path, exploration_order, move_cache, L)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(30)
    pygame.quit()

E_WIDTH, E_HEIGHT, N_WIDTH, N_HEIGHT = 50, 150, 50, 150
P_WIDTH, P_HEIGHT, M_WIDTH, M_HEIGHT = 50, 150, 50, 150
SIX_WIDTH, SIX_HEIGHT, ONE_WIDTH, ONE_HEIGHT = 50, 150, 30, 150

Obstacles = [
    EObstacle((80, 50), E_WIDTH, E_HEIGHT),
    NObstacle((150, 50), N_WIDTH, N_HEIGHT),
    PObstacle((220, 50), P_WIDTH, P_HEIGHT),
    MObstacle((300, 50), M_WIDTH, M_HEIGHT),
    SixObstacle((370, 50), SIX_WIDTH, SIX_HEIGHT),
    SixObstacle((440, 50), SIX_WIDTH, SIX_HEIGHT),
    OneObstacle((510, 50), ONE_WIDTH, ONE_HEIGHT)
]

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
        start_theta = int(input("Enter start orientation (angle in degrees, 0-360): "))
        if 0 <= start_theta < 360:
            valid = True
        else:
            print("Invalid orientation. Please enter an angle between 0 and 360 degrees.")
    valid = False
    while not valid:
        goal_x, goal_y = map(int, input("Enter goal point (x y): ").split())
        if is_valid_point(goal_x, goal_y, Obstacles):
            valid = True
        else:
            print("Invalid goal point. Ensure it is within bounds and not inside an obstacle.")
    valid = False
    while not valid:
        try:
            L = float(input("Enter the length of each step (L): "))
            if L <= 0:
                print("Length must be positive.")
            else:
                valid = True
        except ValueError:
            print("Invalid input. Please enter a numeric value for length.")
except ValueError:
    print("Invalid input. Please enter integer coordinates.")
    sys.exit(1)

start = (start_x, start_y, start_theta)
goal = (goal_x, goal_y, 0)

path, closed_set, exploration_order = astar_search(start, goal, Obstacles, L)
print(len(exploration_order))

move_cache = {angle: (math.cos(math.radians(angle)), math.sin(math.radians(angle))) 
              for angle in range(0, 360, ANGLE_RESOLUTION)}
visualize_astar(Obstacles, path, exploration_order, move_cache, L)

np_path = np.array(path)
np_closed_set = np.array(list(closed_set), dtype=float)

num_visited = np_closed_set.shape[0]
path_length = len(path)
num_frames = num_visited + path_length
