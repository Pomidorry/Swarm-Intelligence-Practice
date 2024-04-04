import numpy as np
import matplotlib.pyplot as plt

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def read_tsp_file(filename):
    points = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("NODE_COORD_SECTION"):
                break
        for line in file:
            if line.strip() == "EOF":
                break
            idx, x, y = map(float, line.split())
            points[int(idx)] = (x, y)
    return points

def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf
    all_paths = []
    
    for iteration in range(n_iterations):
        paths = []
        path_lengths = []
        
        for ant in range(n_ants):
            visited = [False]*n_points
            current_point = np.random.randint(1, n_points+1)  # Start from a random point
            visited[current_point-1] = True
            path = [current_point]
            path_length = 0
            
            while False in visited:
                unvisited = [i for i in range(1, n_points+1) if not visited[i-1]]
                probabilities = np.zeros(len(unvisited))
                
                for i, unvisited_point in enumerate(unvisited):
                    probabilities[i] = pheromone[current_point-1, unvisited_point-1]**alpha / distance(points[current_point], points[unvisited_point])**beta
                
                total_probability = np.sum(probabilities)
                if total_probability == 0:
                    probabilities = np.ones(len(unvisited)) / len(unvisited)  # Равномерное распределение вероятностей, если сумма равна 0
                else:
                    probabilities /= total_probability
                
                next_point = np.random.choice(unvisited, p=probabilities)
                path.append(next_point)
                path_length += distance(points[current_point], points[next_point])
                visited[next_point-1] = True
                current_point = next_point
            
            paths.append(path)
            path_lengths.append(path_length)
            
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length
        
        pheromone *= evaporation_rate
        
        for path, path_length in zip(paths, path_lengths):
            for i in range(n_points-1):
                pheromone[path[i]-1, path[i+1]-1] += Q/path_length
            pheromone[path[-1]-1, path[0]-1] += Q/path_length
        
        all_paths.extend(paths)
    
    plt.figure(figsize=(8, 8))
    
    # Plot points
    plt.scatter([point[0] for point in points.values()], [point[1] for point in points.values()], c='r', marker='o')
    
    
    # Plot best path
    for i in range(n_points-1):
        plt.plot([points[best_path[i]][0], points[best_path[i+1]][0]],
                 [points[best_path[i]][1], points[best_path[i+1]][1]],
                 c='g', linestyle='-', linewidth=2, marker='o')
    plt.plot([points[best_path[0]][0], points[best_path[-1]][0]],
             [points[best_path[0]][1], points[best_path[-1]][1]],
             c='g', linestyle='-', linewidth=2, marker='o')
    
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    plt.grid(True)
    plt.show()

# Example usage:
filename = "E:\\lab4SI\\xqg237.tsp"
points = read_tsp_file(filename)
ant_colony_optimization(points, n_ants=17, n_iterations=200, alpha=1, beta=1, evaporation_rate=0.5, Q=1)
