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
        """
        This function Calculates and returns the Euclidean distance between the position of the given node and the goal node.

        Parameters:
            node (Pathfinding.Node): A node in the grid.
            goal (Pathfinding.Node): The goal node.
        Returns:
            The heuristic (Euclidean distance) from the given node to the goal node.
            Calculates and returns the Euclidean distance between the position of the given node and the goal node.
        """
        return ((node.position[0] - goal.position[0])**2 + (node.position[1] - goal.position[1])**2)**0.5

    def distance(self, node1, node2):
        """
        Calculates and returns the Euclidean distance between the positions of two given nodes.

        Parameters:
            node1 (Pathfinding.Node): First node.
            node2 (Pathfinding.Node): Second node.
        Returns:
            The Euclidean distance between the positions of the two nodes.
        """
        return ((node1.position[0] - node2.position[0])**2 + (node1.position[1] - node2.position[1])**2)**0.5

    def create_nodes(self):
        """
        Creates a 2D grid of nodes based on the specified dimensions and assigns neighbors to each node based on a neighborhood relationship.
        """
        self.nodes = [[self.Node((i * self.pixel_size, j * self.pixel_size), self.WHITE) for j in range(self.height // self.pixel_size)] for i in range(self.width // self.pixel_size)]

        # Assign neighbors
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                for ni in range(max(0, i - 1), min(len(self.nodes), i + 2)):
                    for nj in range(max(0, j - 1), min(len(self.nodes[i]), j + 2)):
                        if i != ni or j != nj:
                            self.nodes[i][j].neighbors.append(self.nodes[ni][nj])


    def astar(self, start, goal):
        """Implements the A* algorithm to find the shortest path from the start node to the goal node.
            Updates the Pygame window to visualize the process.

        Parameters:
            start (Pathfinding.Node): The starting node.
            goal (Pathfinding.Node): The goal node.
        Returns:
            A list of coordinates representing the shortest path from the start to the goal.
        """
        open_set = []
        closed_set = set()

        start.g = 0  # Initialize g for the start node

        heapq.heappush(open_set, start)

        while open_set:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            current_node = heapq.heappop(open_set)

            if current_node == goal:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

            closed_set.add(current_node)
            current_node.color = self.RED
            current_node.draw(self.screen, self.pixel_size)
            pygame.display.update()

            pygame.time.delay(50)

            for neighbor in current_node.neighbors:
                if neighbor in closed_set:
                    continue

                tentative_g = current_node.g + self.distance(current_node, neighbor)

                if tentative_g < neighbor.g:
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = current_node

                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)
                        neighbor.color = self.GREEN
                        neighbor.draw(self.screen, self.pixel_size)
                        pygame.display.update()
        return None

    def draw_path(self, path):
        """            
        Draws the grid, sub-destinations, and the final path on the Pygame window.
        Path is represented by blue lines.

        Parameters:
            path (list): List of coordinates representing the final path.
        """

        for i in range(len(self.nodes)):
            for j in range(len(self.nodes[i])):
                self.nodes[i][j].draw(self.screen, self.pixel_size)

        if path:
            path_coordinates = [(node[0], node[1]) for node in path]  # Extract coordinates from nodes
            pygame.draw.lines(self.screen, self.BLUE, False, path_coordinates, 2)

        pygame.display.update()

    def move_ball_along_path(self, path):
        """
        This function Animates the movement of a ball along the final path on the Pygame window.
        Updates the Pygame window to visualize the moving ball.Prints the coordinates of the ball.

        Parameters:
            path (list): List of coordinates representing the final path.
        """
        ball_position = path[0]
        ball_radius = 8

        index = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if index < len(path):
                ball_position = path[index]
                index += 1

            self.screen.fill(self.WHITE)  # Clear the screen
            pygame.draw.lines(self.screen, self.BLUE, False, path, 2)  # Redraw the path
            pygame.draw.circle(self.screen, self.BLACK, ball_position, ball_radius)  # Draw the moving ball

            # Print the coordinates of the ball
            font = pygame.font.Font(None, 36)
            text = font.render(f"Coordinates: {ball_position}", True, self.BLACK)
            self.screen.blit(text, (10, 10))

            pygame.display.update()
            pygame.time.delay(500)  
        




def main():
    pygame.init()
    width, height = 800, 600

    # Create an instance of the Pathfinding class
    pathfinding = Pathfinding(width, height)

    # Create nodes
    pathfinding.create_nodes()
        # Example starting and ending points
    start_point = [308, 355]
    end_point = [100, 100]

    # Find the closest nodes to the starting and ending points
    start_node = pathfinding.nodes[start_point[0] // pathfinding.pixel_size][start_point[1] // pathfinding.pixel_size]
    end_node = pathfinding.nodes[end_point[0] // pathfinding.pixel_size][end_point[1] // pathfinding.pixel_size]


    # start = pathfinding.nodes[10][10]
    # goal = pathfinding.nodes[0][35]
    # print(start)

    # Find the path passing through the closest sub-destination
    path = pathfinding.astar(start_node, end_node)

    # Draw grid, sub-destinations, and path
    pathfinding.draw_path(path)

    # Move ball along the path and visualize
    pathfinding.move_ball_along_path(path)
    
    pygame.quit()

if __name__ == "__main__":
    main()
