"""Simple aatar algorithm
"""
# import pygame
# import math
# import heapq

# # Constants
# WIDTH, HEIGHT = 700, 700
# ROWS, COLS = HEIGHT // 16, WIDTH // 16
# GRID_SIZE = WIDTH // COLS

# # Colors
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
# RED = (255, 0, 0)
# GREEN = (0, 255, 0)
# BLUE = (0, 0, 255)

# # A* algorithm
# def heuristic(a, b):
#     return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

# def astar(start, end, obstacles):
#     open_set = []
#     closed_set = set()
#     heapq.heappush(open_set, (0, start))
#     came_from = {}

#     g_score = {pos: float('inf') for pos in obstacles}
#     g_score[start] = 0

#     f_score = {pos: float('inf') for pos in obstacles}
#     f_score[start] = heuristic(start, end)

#     while open_set:
#         current = heapq.heappop(open_set)[1]

#         if current == end:
#             path = reconstruct_path(came_from, end)
#             return path

#         closed_set.add(current)

#         for neighbor in neighbors(current):
#             if neighbor in closed_set or neighbor in obstacles:
#                 continue

#             tentative_g_score = g_score[current] + heuristic(current, neighbor)

#             if tentative_g_score < g_score.get(neighbor, float('inf')):
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
#                 heapq.heappush(open_set, (f_score[neighbor], neighbor))

#     return None

# def neighbors(pos):
#     x, y = pos
#     return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

# def reconstruct_path(came_from, current):
#     path = []
#     while current in came_from:
#         path.insert(0, current)
#         current = came_from[current]
#     return path

# # Visualization
# def draw_grid(screen):
#     for i in range(0, WIDTH, GRID_SIZE):
#         pygame.draw.line(screen, WHITE, (i, 0), (i, HEIGHT))
#     for j in range(0, HEIGHT, GRID_SIZE):
#         pygame.draw.line(screen, WHITE, (0, j), (WIDTH, j))

# def draw_obstacles(screen, obstacles):
#     for obstacle in obstacles:
#         pygame.draw.rect(screen, WHITE, (obstacle[0]*GRID_SIZE, obstacle[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

# def draw_path(screen, path):
#     for pos in path:
#         pygame.draw.rect(screen, BLUE, (pos[0]*GRID_SIZE, pos[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

# def draw_start_end(screen, start, end):
#     pygame.draw.rect(screen, GREEN, (start[0]*GRID_SIZE, start[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
#     pygame.draw.rect(screen, RED, (end[0]*GRID_SIZE, end[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

# def run_astar_visualization(start, end, obstacles):
#     pygame.init()
#     screen = pygame.display.set_mode((WIDTH, HEIGHT))
#     pygame.display.set_caption("A* Algorithm Visualization")

#     running = True
#     path = astar(start, end, obstacles)

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False

#         screen.fill(BLACK)
#         draw_grid(screen)
#         draw_obstacles(screen, obstacles)
#         draw_path(screen, path)
#         draw_start_end(screen, start, end)

#         pygame.display.flip()

#     pygame.quit()

# if __name__== "__main__":
#     start_position = (2, 2)
#     end_position = (27, 27)
#     obstacle_positions = [(9,2),(18, 9), (11, 10), (12, 10), (13, 10), (14, 10), (24, 14), (25, 14),
#                            (16, 11), (16, 12), (16, 13), (16, 14), (16, 15), (16, 16), (16, 17),
#                            (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24),
#                            (16, 25), (16, 26), (16, 27)]


"""A star algorithm expanding in for finding path and increasing the size of obstacles as needed
"""
import pygame
import math
import heapq

# Constants
WIDTH, HEIGHT = 800, 600
ROWS, COLS = HEIGHT // 16, WIDTH // 16
GRID_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# A* algorithm
def heuristic(a, b):
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

def astar(start, end, obstacles, obstacle_size):
    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0, start))
    came_from = {}

    g_score = {pos: float('inf') for pos in obstacles}
    g_score[start] = 0

    f_score = {pos: float('inf') for pos in obstacles}
    f_score[start] = heuristic(start, end)

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == end:
            path = reconstruct_path(came_from, end)
            return path

        closed_set.add(current)

        for neighbor in neighbors(current, obstacles, obstacle_size):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if tentative_g_score < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    print("No valid path found.")
    return None

def neighbors(pos, obstacles, obstacle_size):
    x, y = pos
    possible_neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    valid_neighbors = []

    for neighbor in possible_neighbors:
        # Check if the neighbor is within the grid boundaries
        if 0 <= neighbor[0] < COLS and 0 <= neighbor[1] < ROWS:
            # Check if the neighbor is within the expanded obstacle bounds
            if not any(obstacle[0] - obstacle_size + 1 <= neighbor[0] <= obstacle[0] + obstacle_size - 1 and
                       obstacle[1] - obstacle_size + 1 <= neighbor[1] <= obstacle[1] + obstacle_size - 1 for obstacle in obstacles):
                valid_neighbors.append(neighbor)

    return valid_neighbors

def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.insert(0, current)
        current = came_from[current]
    return path

# Visualization
def draw_grid(screen):
    for i in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(screen, WHITE, (i, 0), (i, HEIGHT))
    for j in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(screen, WHITE, (0, j), (WIDTH, j))

def draw_obstacles(screen, obstacles, obstacle_size):
    for obstacle in obstacles:
        x, y = obstacle
        for _ in range(obstacle_size):
            for _ in range(obstacle_size):
                pygame.draw.rect(screen, WHITE, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                x += 1
            x = obstacle[0]
            y += 1
            x_offset = 1  # Initialize the offset for the left side
            for _ in range(obstacle_size - 1):
                pygame.draw.rect(screen, WHITE, ((x - x_offset) * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                x_offset += 1
            y_offset = 1  # Initialize the offset for the top side
            for _ in range(obstacle_size - 1):
                pygame.draw.rect(screen, WHITE, (x * GRID_SIZE, (y - y_offset) * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                y_offset += 1
                x_offset_top = 1  # Initialize the offset for the top right side
                for _ in range(obstacle_size - 1):
                    pygame.draw.rect(screen, WHITE, ((x + x_offset_top) * GRID_SIZE, (y - y_offset) * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                    x_offset_top += 1
def draw_path(screen, path):
    if path is not None:
        for pos in path:
            pygame.draw.rect(screen, BLUE, (pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

def draw_start_end(screen, start, end):
    pygame.draw.rect(screen, GREEN, (start[0] * GRID_SIZE, start[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
    pygame.draw.rect(screen, RED, (end[0] * GRID_SIZE, end[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))


def run_astar_visualization(start, end, obstacles, obstacle_size):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("A* Algorithm Visualization")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        draw_grid(screen)
        draw_obstacles(screen, obstacles, obstacle_size)
        
        path = astar(start, end, obstacles, obstacle_size)
        draw_path(screen, path)
        
        draw_start_end(screen, start, end)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    start_position = (2, 2)
    end_position = (37, 27)
    obstacle_positions = [(9,2),(18, 9), (24, 14), 
                           ]
    obstacle_size = 2  # Increase obstacle size to 2x2

    run_astar_visualization(start_position, end_position, obstacle_positions, obstacle_size)
