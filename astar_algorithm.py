import pygame
import heapq

class Pathfinding:
    def __init__(self, width, height, pixel_size=16):
        self.width = width
        self.height = height
        self.pixel_size = pixel_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("A* Pathfinding Visualization")

        # Define colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        self.nodes = [[]]
        self.path = []
    class Node:
        def __init__(self, position, color):
            self.position = position
            self.neighbors = []
            self.color = color
            self.g = float('inf')  # Initialize g to infinity
            self.h = 0
            self.f = 0
            self.parent = None

        def draw(self, screen, pixel_size):
            pygame.draw.rect(screen, self.color, (self.position[0], self.position[1], pixel_size, pixel_size))

        def __lt__(self, other):
            return self.f < other.f
    

    def heuristic(self, node, goal):
        return ((node.position[0] - goal.position[0])**2 + (node.position[1] - goal.position[1])**2)**0.5

    def distance(self, node1, node2):
        return ((node1.position[0] - node2.position[0])**2 + (node1.position[1] - node2.position[1])**2)**0.5

    def create_nodes(self):
        self.nodes = [[self.Node((i * self.pixel_size, j * self.pixel_size), self.WHITE) for j in range(self.height // self.pixel_size)] for i in range(self.width // self.pixel_size)]

        # Assign neighbors
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                for ni in range(max(0, i - 1), min(len(self.nodes), i + 2)):
                    for nj in range(max(0, j - 1), min(len(self.nodes[i]), j + 2)):
                        if i != ni or j != nj:
                            self.nodes[i][j].neighbors.append(self.nodes[ni][nj])
