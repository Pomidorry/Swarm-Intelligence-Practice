import numpy as np
import matplotlib.pyplot as plt
import random

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def generate_random_points_on_circle(radius, center, num_points):
    """
    Generate random points on the perimeter of a circle.
    Input:
    1. radius: radius of the circle
    2. center: center coordinates of the circle (x, y)
    3. num_points: number of points to generate
    Output:
    List of random points on the perimeter of the circle
    """
    points = []
    for _ in range(num_points):
        theta = random.uniform(0, 2*np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        points.append((x, y))
    return points

def generate_random_points_within_circle(radius, center, num_points):
    """
    Generate random points within a circle.
    Input:
    1. radius: radius of the circle
    2. center: center coordinates of the circle (x, y)
    3. num_points: number of points to generate
    Output:
    List of random points within the circle
    """
    points = []
    for _ in range(num_points):
        theta = random.uniform(0, 2*np.pi)
        r = radius * np.sqrt(random.uniform(0, 1))
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        points.append((x, y))
    return points

def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf
    
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = np.random.randint(n_points)
            visited[current_point] = True
            path = [current_point]
            path_length = 0
            
            while False in visited:
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))
                
                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point, unvisited_point]**alpha / distance(points[current_point], points[unvisited_point])**beta
                
                total_probability = np.sum(probabilities)
                if total_probability == 0:
                    probabilities = np.ones(len(unvisited)) / len(unvisited)  # Равномерное распределение вероятностей, если сумма равна 0
                else:
                    probabilities /= total_probability
                
                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point] = True
                current_point = next_point
            
            paths.append(path)
            path_lengths.append(path_length)
            
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
        
        pheromone *= evaporation_rate
        
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points-1):
                pheromone[path[i], path[i+1]] += Q/path_length
            pheromone[path[-1], path[0]] += Q/path_length
    
    plt.figure(figsize=(8, 8))
    
    # Plot circle
    circle = plt.Circle((0, 0), radius, color='b', fill=False)
    plt.gca().add_patch(circle)
    
    # Plot points
    plt.scatter([point[0] for point in points], [point[1] for point in points], c='r', marker='o')
    
    # Plot best path
    for i in range(n_points-1):
        plt.plot([points[best_path[i]][0], points[best_path[i+1]][0]],
                 [points[best_path[i]][1], points[best_path[i+1]][1]],
                 c='r', linestyle='-', linewidth=2, marker='o')
    plt.plot([points[best_path[0]][0], points[best_path[-1]][0]],
             [points[best_path[0]][1], points[best_path[-1]][1]],
             c='r', linestyle='-', linewidth=2, marker='o')
    
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.grid(True)
    plt.show()
    
# Example usage:
radius = 10
center = (0, 0)
num_points = 20
points = generate_random_points_on_circle(radius, center, num_points) # Generate 20 random 2D points within a circle
ant_colony_optimization(points, n_ants=15, n_iterations=100, alpha=1, beta=1, evaporation_rate=0.5, Q=1)