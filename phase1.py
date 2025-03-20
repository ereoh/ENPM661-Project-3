import numpy as np
import cv2
from collections import deque
import sys
import heapq

# Define colors
red = (0, 0, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
grey = (128, 128, 128)

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
            (origin[0] + 6, origin[1] + height - 16),
            (origin[0] + width - 6, origin[1] + height - 16),
            (origin[0] + width - 6, origin[1] + height)
        ])
        self.circle = Circle((origin[0] + width - 6, origin[1] + height - 8), 8)

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

# Function to draw the canvas with obstacles
def draw_canvas(obstacles, canvas):
    # Iterate over canvas pixels
    for i in range(canvas.shape[0]):
        for j in range(canvas.shape[1]):
            for obstacle in obstacles:
                # Convert canvas coordinates to obstacle coordinates
                y = 49 - i
                x = j
                # Color pixel if inside obstacle
                if obstacle.is_inside_obstacle(x, y):
                    canvas[i, j] = red

def astar_search(start, goal, obstacles, canvas):
    # Define possible moves and their costs
    moves = [(-1, 0, 1.0), (-1, 1, 1.4), (0, 1, 1.0), (1, 1, 1.4),
             (1, 0, 1.0), (1, -1, 1.4), (0, -1, 1.0), (-1, -1, 1.4)]
    
    # Heuristic function to estimate the cost from current node to goal
    def heuristic(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return min(dx, dy) * 1.4 + abs(dx - dy) * 1.0  # Movement cost heuristic
    
    # Initialize the open set with the start node
    open_set = []
    heapq.heappush(open_set, (0, start))
    C2C = {start: 0}  # Cost from start to current node
    total_cost = {start: heuristic(start, goal)}  # Estimated cost from start to goal
    parent = {}  # To reconstruct the path
    
    frame_buffer = []  # Store frames for smooth playback
    
    while open_set:
        # Get the node with the lowest f_score
        _, current = heapq.heappop(open_set)
        x, y = current
        
        # Mark the current node as visited on the canvas
        if (x, y) != start and (x, y) != goal:
            canvas[50 - y, x - 1] = blue
        
        frame_buffer.append(canvas.copy())
        
        # If the goal is reached, exit the loop
        if current == goal:
            break
        
        # Explore neighbors
        for dx, dy, cost in moves:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            
            # Skip if the new position is out of bounds or inside an obstacle
            if not is_valid_point(new_x, new_y, obstacles):
                continue
            
            # Calculate the tentative C2C for the new position
            tentative_C2C = C2C[current] + cost
            if new_pos not in C2C or tentative_C2C < C2C[new_pos]:
                # Update the parent, C2C, and total_cost for the new position
                parent[new_pos] = current
                C2C[new_pos] = tentative_C2C
                total_cost[new_pos] = tentative_C2C + heuristic(new_pos, goal)
                heapq.heappush(open_set, (total_cost[new_pos], new_pos))
    
     # Backtrack to reconstruct the path
    path = []
    current = goal
    while current in parent:
        path.append(current)
        current = parent[current]
    path.reverse() # Reverse the path to get it from start to goal

    print("Path found!")

    # Smoothly draw the path
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        cv2.line(canvas, (x1, 49 - y1), (x2, 49 - y2), green, 1)
        frame_buffer.append(canvas.copy())

    # add 2 seconds of end frames
    for i in range(120):
        frame_buffer.append(canvas.copy())
    
    # Save the frames as an MP4 file
    height, width, layers = canvas.shape
    video = cv2.VideoWriter('AStar_animation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, (width, height))

    for frame in frame_buffer:
        video.write(frame)

    video.release()
    print("Animation saved as AStar_animation.mp4")

    # Play back the frames animation
    for frame in frame_buffer:
        cv2.imshow("Canvas", frame)
        cv2.waitKey(1)

    return path

def is_valid_point(x, y, obstacles):
    # Check if point is within canvas bounds and not inside any obstacle
    if not (2 <= x <= 177 and 2 <= y <= 47):
        return False
    for obstacle in obstacles:
        if obstacle.is_inside_obstacle(x, y):
            return False
    return True

# Initialize canvas and obstacles
canvas = np.zeros((50, 180, 3), np.uint8)
canvas[:] = grey
E_obstacle = EObstacle((10, 10), 20, 30)
N_obstacle = NObstacle((35, 10), 20, 30)
P_Obstacle = PObstacle((60, 10), 20, 30)
M_Obstacle = MObstacle((85, 10), 20, 30)
Six_Obstacle_1 = SixObstacle((110, 10), 15, 30)
Six_Obstacle_2 = SixObstacle((130, 10), 15, 30)
One_Obstacle = OneObstacle((150, 10), 10, 30)

Obstacles = [E_obstacle, N_obstacle, P_Obstacle, M_Obstacle, Six_Obstacle_1, Six_Obstacle_2, One_Obstacle]

# Draw obstacles on the canvas
draw_canvas(Obstacles, canvas)

# Request start and goal points from the command line
try:
    print("Maze: Bottom left corner is (2, 2) and top right corner is (177, 47)")
    start_x, start_y = map(int, input("Enter start point (x y): ").split())
    goal_x, goal_y = map(int, input("Enter goal point (x y): ").split())
except ValueError:
    print("Invalid input. Please enter integer coordinates.")
    sys.exit(1)

# Validate start and goal points
if not is_valid_point(start_x, start_y, Obstacles):
    print("Invalid start point. It is either out of bounds or inside an obstacle.")
    sys.exit(1)
if not is_valid_point(goal_x, goal_y, Obstacles):
    print("Invalid goal point. It is either out of bounds or inside an obstacle.")
    sys.exit(1)

start = (start_x, start_y)
goal = (goal_x, goal_y)

# Color the start and goal points green on the canvas
canvas[49 - start_y, start_x] = green
canvas[49 - goal_y, goal_x] = green

# Perform BFS search and get the path
path = astar_search(start, goal, Obstacles, canvas)

# Display the final canvas
cv2.imshow("Canvas", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()