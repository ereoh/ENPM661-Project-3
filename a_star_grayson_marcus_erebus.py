import numpy as np
from collections import deque
import sys
import heapq
import math
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.animation as animation
import matplotlib
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Define colors
red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
grey = (128, 128, 128)

# define space resolution
ANGLE_RESOLUTION = 30
STEP_RESOLUTION = 1

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
            (origin[0] + 6, origin[1] + height - 6),
            (origin[0] + width - 6, origin[1]),
            (origin[0] + width - 6, origin[1] + 6)
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
            (origin[0] + 6, origin[1] + height - 6),
            (origin[0] + width // 2, origin[1] + height // 2),
            (origin[0] + width // 2, origin[1] + height // 2 + 6)
        ])
        self.right_diagonal = LineDefinedObstacle([
            (origin[0] + width - 6, origin[1] + height),
            (origin[0] + width - 6, origin[1] + height - 6),
            (origin[0] + width // 2, origin[1] + height // 2),
            (origin[0] + width // 2, origin[1] + height // 2 + 6)
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

def astar_search(start, goal, obstacles, L):
    move_angles = [-60, -30, 0, 30, 60]
    move_cache = {angle: (L * math.cos(math.radians(angle)),
                          L * math.sin(math.radians(angle)))
                  for angle in range(0, 360, ANGLE_RESOLUTION)}
    
    start_node = (*start,)
    goal_pos = goal[:2]
    
    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    open_set = []
    start_key = discretize(*start_node)
    heapq.heappush(open_set, (0, start_node))
    C2C = {start_key: 0}
    total_cost = {start_key: heuristic(start, goal_pos)}
    parent = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        x, y, theta = current
        key = discretize(x, y, theta)

        if key in visited:
            continue
        visited.add(key)

        if heuristic((x, y), goal_pos) < 0.5:
            goal_node = current
            break

        for delta_angle in move_angles:
            new_theta = (theta + delta_angle) % 360
            move_theta = discretize(0, 0, new_theta)[2]
            dx, dy = move_cache[move_theta]
            new_x, new_y = x + dx, y + dy
            new_key = discretize(new_x, new_y, new_theta)

            if new_key in visited or not is_valid_point(new_x, new_y, obstacles):
                continue

            tentative_C2C = C2C[key] + L
            if new_key not in C2C or tentative_C2C < C2C[new_key]:
                C2C[new_key] = tentative_C2C
                total_cost[new_key] = tentative_C2C + heuristic((new_x, new_y), goal_pos)
                heapq.heappush(open_set, (total_cost[new_key], (new_x, new_y, new_theta)))
                parent[new_key] = key

    # Backtrack using discretized keys
    path = []
    key = discretize(*goal_node)
    while key in parent:
        x, y, theta = key
        path.append((x, y, theta))
        key = parent[key]
    path.reverse()

    print("Path found!")
    return path, visited

def is_valid_point(x, y, obstacles):
    # Check if point is within canvas bounds and not inside any obstacle
    if not (5 <= x <= (WIDTH - 6) and 5 <= y <= (HEIGHT - 6)):
        return False
    for obstacle in obstacles:
        if obstacle.is_inside_obstacle(x, y):
            return False
    return True

## Animation Functions

def init_animation(start, goal, path, obstacles, num_visited):
    print("Initializing animation")

    # draw background
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(f'A-Star Search from {start} to {goal}')
    ax.set_xlim(-10, WIDTH + 10)
    ax.set_ylim(-10, HEIGHT + 10)
    # Draw borders (Blue)
    border_x = [4, 4, WIDTH-5, WIDTH-5, 4]
    border_y = [4, HEIGHT-5, HEIGHT-5, 4, 4]
    ax.plot(border_x, border_y, linewidth=2, c='b')

    # draw obstacles (Blue)
    for i in range(WIDTH):
        for j in range(HEIGHT):
            for obstacle in obstacles:
                # Color pixel if inside obstacle
                if obstacle.is_inside_obstacle(i, j):
                    ax.plot(i, j, marker='s', color='blue')

    # draw start (Green)
    ax.scatter(start[0], start[1], marker='s', c='#7dffa0')

    # draw goal (Red)
    ax.scatter(goal[0], goal[1], marker='s', c='red')

    # Init exploration and path artists
    # arrows
    '''
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    ax.quiver(
        [X, Y], # arrow locations
        U, # x arrow vector from location
        V, # y arrow vector from location
        [C], # colormapping
        angles = 'xy', # vector is from (x,y) -> (x+u, y+v)
        scale = float, # scales length of arrow
    )
    '''
    
    exploration_draw = ax.scatter([], [], marker='s', c=[], cmap='viridis')
    exploration_draw.set_clim(0, num_visited) # colorbar init
    path_line, = ax.plot([], [], marker='s', linewidth=1, c='#ff29f8')

    # Init Colorbar
    cstep = max(1, int(num_visited / 10)) # want 10 ticks along colorbar
    cbar = fig.colorbar(exploration_draw, ax=ax)
    cbar.set_label('Explored Order')

    filename = f"AStar_animation_{start[0]}-{start[1]}_to_{goal[0]}-{goal[1]}"

    return fig, exploration_draw, path_line, cbar, cstep, filename

def update_animation(i):
    # Q.set_UVC(U, V)
    # first draw exploration
    if i < num_visited:
        exploration_draw.set_offsets(np_closed_set[:i])
        exploration_draw.set_array(np.arange(i))
        cbar.set_ticks(np.arange(0, np_closed_set.shape[0], cstep))

        path_line.set_data([],[])
    # then path
    else:
        idx = i - np_closed_set.shape[0]
        path_line.set_data(np_path[:idx].T)

    return exploration_draw, path_line,

def save_update(i, total):
    save_progress.update(1)

def create_animation(fig, num_frames, filename, show=True, write=True):
    global save_progress

    # generate animation
    ani = FuncAnimation(
        fig, # figure to animate on
        update_animation, # call each frame
        frames=num_frames,  # total number of frames in animation
        # init_func = None, # draw clear frame
        # returns iterable of artists
        interval=1, # delay between frames in ms
        blit=True, # use blitting optimization
    )

    if show:
        plt.show()

    if write:

        save_progress = tqdm(total = num_frames, desc = "Saving Animation", unit='frames')
        # save as MP4 or GIF
        available_writers = animation.writers.list()
        available_writers = animation.writers.list()

        writer = 'ffmpeg'

        if 'ffmpeg' in available_writers:
            filename += ".mp4"
        else:
            filename += ".gif"
            writer = "pillow"

        # write animation as gif to disk
        if writer == 'pillow':
            print("Warning: ffmpeg not found! Using pillow and saving as GIF. This is significantly slower.")
        
        ani.save(
            filename, 
            writer=writer, 
            fps=60, 
            progress_callback=save_update
        )
        print(f"Saved animation to {filename}")

WIDTH = 600
HEIGHT = 250
E_obstacle = EObstacle((80, 50), 50, 150)
N_obstacle = NObstacle((150, 50), 50, 150)
P_Obstacle = PObstacle((220, 50), 50, 150)
M_Obstacle = MObstacle((300, 50), 50, 150)
Six_Obstacle_1 = SixObstacle((370, 50), 50, 150)
Six_Obstacle_2 = SixObstacle((440, 50), 50, 150)
One_Obstacle = OneObstacle((510, 50), 30, 150)

Obstacles = [E_obstacle, N_obstacle, P_Obstacle, M_Obstacle, Six_Obstacle_1, Six_Obstacle_2, One_Obstacle]


# Request start and goal points from the command line
try:
    print("Maze: Bottom left corner is ( 5 , 5 ) and top right corner is (", (WIDTH - 6), ",", (HEIGHT - 6),")")

    valid = False
    while not valid:
        start_x, start_y = map(int, input("Enter start point (x y): ").split())
        if is_valid_point(start_x, start_y, Obstacles):
            valid = True
        else:
            print("Invalid start point. It is either out of bounds or inside an obstacle.")
    
    valid = False
    while not valid:
        goal_x, goal_y = map(int, input("Enter goal point (x y): ").split())
        if is_valid_point(goal_x, goal_y, Obstacles):
            valid = True
        else:
            print("Invalid goal point. It is either out of bounds or inside an obstacle.")
except ValueError:
    print("Invalid input. Please enter integer coordinates.")
    sys.exit(1)

start = (start_x, start_y, 0)
goal = (goal_x, goal_y, 0)

# Perform A Star search and get the path
path, closed_set = astar_search(start, goal, Obstacles, 1)

# convert to numpy arrays for animation
np_path = np.array(path)
np_closed_set = np.array(list(closed_set), dtype=float)

num_visited = np_closed_set.shape[0]
path_length = len(path)
num_frames = num_visited + path_length

# Animate exploration and path
fig, exploration_draw, path_line, cbar, cstep, filename = init_animation(start, goal, path, Obstacles, num_visited)

# Save animation to disk
save_progress = None
create_animation(fig, num_frames, filename, show = False, write = True)