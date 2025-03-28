import heapq
import math
import pygame
import time

# Define colors for visualization
red, blue, green, grey = (0, 0, 255), (255, 0, 0), (0, 255, 0), (128, 128, 128)

# Base class for obstacles
class Obstacle:
    def is_inside_obstacle(self, x, y):
        raise NotImplementedError  # To be implemented by subclasses

# Obstacle defined by lines connecting vertices
class LineDefinedObstacle:
    def __init__(self, vertices):
        self.lines = self.compute_line_constraints(vertices)  # Precompute line constraints

    def compute_line_constraints(self, vertices):
        lines = []
        for i in range(len(vertices)):
            # Compute line equation ax + by + c = 0
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            a, b, c = y2 - y1, x1 - x2, -(y2 - y1) * x1 - (x1 - x2) * y1
            # Determine which side of the line is inside the obstacle
            side = 1 if (a * vertices[(i + 2) % len(vertices)][0] + b * vertices[(i + 2) % len(vertices)][1] + c) > 0 else -1
            lines.append((a, b, c, side))
        return lines

    def is_inside_obstacle(self, x, y):
        # Check if the point satisfies all line constraints
        return all((a * x + b * y + c) * side >= 0 for a, b, c, side in self.lines)

# Circular obstacle
class Circle(Obstacle):
    def __init__(self, center, radius):
        self.center, self.radius = center, radius  # Store center and radius

    def is_inside_obstacle(self, x, y):
        # Check if the point is within the circle
        return (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2 <= self.radius ** 2

# Obstacle shaped like the letter 'E'
class EObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
        # Define the bars and spine of the 'E'
        self.bottom_bar = LineDefinedObstacle([
            (origin[0] + 16, origin[1]), 
            (origin[0] + width, origin[1]), 
            (origin[0] + width, origin[1] + 16), 
            (origin[0] + 16, origin[1] + 16)
        ])
        self.top_bar = LineDefinedObstacle([
            (origin[0] + 16, origin[1] + height - 16), 
            (origin[0] + width, origin[1] + height - 16), 
            (origin[0] + width, origin[1] + height), 
            (origin[0] + 16, origin[1] + height)
        ])
        self.middle_bar = LineDefinedObstacle([
            (origin[0] + 16, (origin[1] + height // 2) - 8), 
            (origin[0] + width, (origin[1] + height // 2) - 8), 
            (origin[0] + width, (origin[1] + height // 2) + 8), 
            (origin[0] + 16, (origin[1] + height // 2) + 8)
        ])
        self.spine = LineDefinedObstacle([
            (origin[0], origin[1]), 
            (origin[0] + 16, origin[1]), 
            (origin[0] + 16, origin[1] + height), 
            (origin[0], origin[1] + height)
        ])

    def is_inside_obstacle(self, x, y):
        # Check if the point is inside any part of the 'E'
        return (self.bottom_bar.is_inside_obstacle(x, y) or 
                self.top_bar.is_inside_obstacle(x, y) or 
                self.middle_bar.is_inside_obstacle(x, y) or 
                self.spine.is_inside_obstacle(x, y))

# Obstacle shaped like the letter 'N'
class NObstacle(Obstacle): 
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
        # Define the bars and diagonal of the 'N'
        self.left_bar = LineDefinedObstacle([
            (origin[0], origin[1]), 
            (origin[0] + 16, origin[1]), 
            (origin[0] + 16, origin[1] + height), 
            (origin[0], origin[1] + height)
        ])
        self.right_bar = LineDefinedObstacle([
            (origin[0] + width - 16, origin[1]), 
            (origin[0] + width, origin[1]), 
            (origin[0] + width, origin[1] + height), 
            (origin[0] + width - 16, origin[1] + height)
        ])
        self.diagonal_bar = LineDefinedObstacle([
            (origin[0] + 16, origin[1] + height),
            (origin[0] + 16, origin[1] + height - 55),
            (origin[0] + width - 16, origin[1]),
            (origin[0] + width - 16, origin[1] + 55)
        ])
    
    def is_inside_obstacle(self, x, y):
        # Check if the point is inside any part of the 'N'
        return (self.left_bar.is_inside_obstacle(x, y) or 
                self.right_bar.is_inside_obstacle(x, y) or 
                self.diagonal_bar.is_inside_obstacle(x, y))

# Obstacle shaped like the letter 'P'
class PObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
        # Define the spine and head of the 'P'
        self.spine = LineDefinedObstacle([
            (origin[0], origin[1]), 
            (origin[0] + 16, origin[1]), 
            (origin[0] + 16, origin[1] + height), 
            (origin[0], origin[1] + height)
        ])
        self.head = LineDefinedObstacle([
            (origin[0] + 16, origin[1] + height),
            (origin[0] + 16, origin[1] + height - 50),
            (origin[0] + width - 16, origin[1] + height - 50),
            (origin[0] + width - 16, origin[1] + height)
        ])
        self.circle = Circle((origin[0] + width - 16, origin[1] + height - 25), 25)

    def is_inside_obstacle(self, x, y):
        # Check if the point is inside any part of the 'P'
        return (self.spine.is_inside_obstacle(x, y) or 
                self.head.is_inside_obstacle(x, y) or 
                self.circle.is_inside_obstacle(x, y))

# Obstacle shaped like the letter 'M'
class MObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
        # Define the bars and diagonals of the 'M'
        self.left_bar = LineDefinedObstacle([
            (origin[0], origin[1]), 
            (origin[0] + 16, origin[1]), 
            (origin[0] + 16, origin[1] + height), 
            (origin[0], origin[1] + height)
        ])
        self.right_bar = LineDefinedObstacle([
            (origin[0] + width - 16, origin[1]), 
            (origin[0] + width, origin[1]), 
            (origin[0] + width, origin[1] + height), 
            (origin[0] + width - 16, origin[1] + height)
        ])
        self.left_diagonal = LineDefinedObstacle([
            (origin[0] + 16, origin[1] + height),
            (origin[0] + 16, origin[1] + height - 55),
            (origin[0] + width // 2, origin[1] + height // 2),
            (origin[0] + width // 2, origin[1] + height // 2 + 55)
        ])
        self.right_diagonal = LineDefinedObstacle([
            (origin[0] + width - 16, origin[1] + height),
            (origin[0] + width - 16, origin[1] + height - 55),
            (origin[0] + width // 2, origin[1] + height // 2),
            (origin[0] + width // 2, origin[1] + height // 2 + 55)
        ])
    
    def is_inside_obstacle(self, x, y):
        # Check if the point is inside any part of the 'M'
        return (self.left_bar.is_inside_obstacle(x, y) or 
                self.right_bar.is_inside_obstacle(x, y) or 
                self.left_diagonal.is_inside_obstacle(x, y) or 
                self.right_diagonal.is_inside_obstacle(x, y))

# Obstacle shaped like the number '6'
class SixObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
        # Define the base, spine, and top bar of the '6'
        self.base = LineDefinedObstacle([
            (origin[0], origin[1]),
            (origin[0] + width, origin[1]),
            (origin[0] + width, origin[1] + height // 2),
            (origin[0], origin[1] + height // 2)
        ])
        self.spine = LineDefinedObstacle([
            (origin[0], origin[1] + height // 2),
            (origin[0] + 16, origin[1] + height // 2),
            (origin[0] + 16, origin[1] + height),
            (origin[0], origin[1] + height)
        ])
        self.top_bar = LineDefinedObstacle([
            (origin[0] + 16, origin[1] + height - 16),
            (origin[0] + width, origin[1] + height - 16),
            (origin[0] + width, origin[1] + height),
            (origin[0] + 16, origin[1] + height)
        ])

    def is_inside_obstacle(self, x, y):
        # Check if the point is inside any part of the '6'
        return (self.top_bar.is_inside_obstacle(x, y) or 
                self.base.is_inside_obstacle(x, y) or 
                self.spine.is_inside_obstacle(x, y))

# Obstacle shaped like the number '1'
class OneObstacle(Obstacle):
    def __init__(self, origin, width, height):
        self.origin, self.width, self.height = origin, width, height
        # Define the bar of the '1'
        self.bar = LineDefinedObstacle([
            (origin[0], origin[1]),
            (origin[0] + width, origin[1]),
            (origin[0] + width, origin[1] + height),
            (origin[0], origin[1] + height)
        ])
    
    def is_inside_obstacle(self, x, y):
        # Check if the point is inside the '1'
        return self.bar.is_inside_obstacle(x, y)

# Resolution and map dimensions
WIDTH, HEIGHT, SCALE = 600, 250, 3

# Define movement directions and discretization parameters
DIRECTIONS = [-60, -30, 0, 30, 60]  # Degrees
THETA_DISCRETIZATION = 30           # Degrees
POS_DISCRETIZATION = 0.5            # Units

def is_valid_point(x, y, obstacles):
    # Check if the point is within bounds and not inside any obstacle
    if not (5 <= x <= WIDTH - 6 and 5 <= y <= HEIGHT - 6):
        return False
    return not any(obstacle.is_inside_obstacle(x, y) for obstacle in obstacles)

def normalize_angle(theta):
    # Normalize angle to the range [0, 360)
    return (theta + 360) % 360

def discretize_state(x, y, theta):
    # Convert continuous state to discrete state for grid-based representation
    return (round(x / (POS_DISCRETIZATION * L)), round(y / (POS_DISCRETIZATION * L)), round(normalize_angle(theta) / THETA_DISCRETIZATION))

def heuristic(x, y, goal):
    # Compute Euclidean distance as the heuristic
    return math.hypot(goal[0] - x, goal[1] - y)

def astar_search(start, goal, obstacles, L):
    # Initialize start states
    start_x, start_y, start_theta = start
    start_theta = normalize_angle(start_theta)
    start_state = (start_x, start_y, start_theta)
    start_discrete = discretize_state(*start_state)

    # Initialize visited matrix
    VisitedMatrix = [[[0 for _ in range(int(360 / THETA_DISCRETIZATION))] 
                        for _ in range(int(HEIGHT / (POS_DISCRETIZATION * L)))] 
                        for _ in range(int(WIDTH / (POS_DISCRETIZATION * L)))]

    # Priority queue for A* search
    open_set = [(heuristic(start_x, start_y, goal), 0, start_state)]
    came_from = {}  # Track the path
    cost_so_far = {start_discrete: 0}  # Cost to reach each state
    explored_order = []  # Track exploration order

    while open_set:
        # Get the state with the lowest cost
        _, cost, (x, y, theta) = heapq.heappop(open_set)
        current_discrete = discretize_state(x, y, theta)
        explored_order.append((x, y, theta))

        if heuristic(x, y, goal) <= 1.5 * L:
            # Reconstruct path if goal is reached
            path = [(x, y, theta)]
            while current_discrete in came_from:
                current_discrete = came_from[current_discrete]
                x_d, y_d, t_d = current_discrete
                path.append((x_d * POS_DISCRETIZATION * L,
                             y_d * POS_DISCRETIZATION * L,
                             t_d * THETA_DISCRETIZATION))
            path.reverse()
            return path, explored_order

        for d_theta in DIRECTIONS:
            # Compute new state after applying action
            new_theta = normalize_angle(theta + d_theta)
            rad = math.radians(new_theta)
            new_x = x + L * math.cos(rad)
            new_y = y + L * math.sin(rad)

            # Check if the new point is valid
            if not is_valid_point(new_x, new_y, obstacles):
                continue

            new_discrete = discretize_state(new_x, new_y, new_theta)
            i, j, k = new_discrete
            new_cost = cost + L
            # Check if the new state has been visited or if the cost is lower
            if (VisitedMatrix[i][j][k] == 0) or (new_cost < cost_so_far.get(new_discrete, float('inf'))):
                VisitedMatrix[i][j][k] = 1  # mark as visited region
                cost_so_far[new_discrete] = new_cost
                priority = new_cost + heuristic(new_x, new_y, goal)
                heapq.heappush(open_set, (priority, new_cost, (new_x, new_y, new_theta)))
                came_from[new_discrete] = current_discrete  # Track the path

    return None, explored_order  # No path found

def draw_obstacles(screen, obstacles):
    # Draw obstacles on the screen
    for x in range(WIDTH):
        for y in range(HEIGHT):
            if not is_valid_point(x, y, obstacles):
                screen.set_at((int(x * SCALE), int((HEIGHT - y) * SCALE)), grey)

def animate_astar(screen, obstacles, path, exploration_order, move_cache, L):
    # Animate the A* search process
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
        
        pygame.draw.rect(screen, blue, (int(path[-1][0]) * SCALE, (HEIGHT - int(path[-1][1])) * SCALE, 5, 5))
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
    # Visualize the A* search process
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
        clock.tick(60)
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


print("Maze: Bottom left corner is (5, 5) and top right corner is (", (WIDTH - 6), ",", (HEIGHT - 6), ")")
valid = False
while not valid:
    try:
        start_x, start_y = map(int, input("Enter start point (x y): ").split())
        if is_valid_point(start_x, start_y, Obstacles):
            valid = True
        else:
            print("Invalid start point. Ensure it is within bounds and not inside an obstacle.")
    except ValueError:
        print("Invalid input. Please enter integer coordinates for the start point.")

valid = False
while not valid:
    try:
        start_theta = int(input("Enter start orientation (angle in degrees, 0-360): "))
        if 0 <= start_theta < 360:
            valid = True
        else:
            print("Invalid orientation. Please enter an angle between 0 and 360 degrees.")
    except ValueError:
        print("Invalid input. Please enter a numeric value for the angle.")

valid = False
while not valid:
    try:
        goal_x, goal_y = map(int, input("Enter goal point (x y): ").split())
        if is_valid_point(goal_x, goal_y, Obstacles):
            valid = True
        else:
            print("Invalid goal point. Ensure it is within bounds and not inside an obstacle.")
    except ValueError:
        print("Invalid input. Please enter integer coordinates for the goal point.")

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

start = (start_x, start_y, start_theta)
goal = (goal_x, goal_y)

start_time = time.time()  # Start time for the A* search
print("Starting A* search...")
path, exploration_order = astar_search(start, goal, Obstacles, L)
end_time = time.time()  # End time for the A* search
print("A* search completed in {:.4f} seconds".format(end_time - start_time))
# Print the results of the A* search
print(len(exploration_order), " nodes explored during A* search")
if path is None:
    print("No path found to the goal.")
else:
    print("Path found to the goal.")

move_cache = {angle: (math.cos(math.radians(angle)), math.sin(math.radians(angle))) 
              for angle in range(0, 360, THETA_DISCRETIZATION)}
visualize_astar(Obstacles, path, exploration_order, move_cache, L)
