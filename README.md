# CSCI323
This repository discuss about the Travelling Salesman Problem with the usage of 2-opt algorithm and Christofides algorithm. Exploring and comparing these algorithms, then create a merged algorithm that combine both algorithm's advantages to solve the problem in a more balanced way by balancing their runtime and stability.

CSCI323 Assignment 2
Travelling Salesman Problem


2-opt algorithm with 8 points

import networkx as nx
import itertools
import matplotlib.pyplot as plt
import time
import math
import random

# Step 1: Cost function
def cost(route):
    total = 0
    for i in range(len(route) - 1):
        total += distances[(route[i], route[i+1])]
    return total

# Step 2: 2-opt function
def two_opt(route):
    best = route
    improved = True
    cost_history = [cost(best)] # Added for performance curve
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                if cost(new_route) < cost(best):
                    best = new_route
                    improved = True
                    cost_history.append(cost(best)) # Added for performance curve
        route = best
    return best, cost_history

# Step 3: Define city coordinates
city_coords = {
    'A': (0, 0),
    'B': (2, 3),
    'C': (5, 4),
    'D': (6, 1),
    'E': (3, 8),
    'F': (1, 7),
    'G': (9, 5),
    'H': (1, 1)
}

# Step 4: Compute pairwise distances
# Create a new dictionary to keep the distances between cities
distances = {}
# Get a cities list of city_coords keys created
cities = list(city_coords.keys())
# Outer loop to ensure all cities are computed
for i in range(len(cities)):
    # Inner loop, ensuring that each unordered pair is computed once
    for j in range(i + 1, len(cities)):
	      # Assign variable for each pair
        u, v = cities[i], cities[j]
	      # Get the coordinated of u and v
        x1, y1 = city_coords[u]
        x2, y2 = city_coords[v]
	      # Get the distance between u and v
        d = math.hypot(x1 - x2, y1 - y2)
	      # Store them in bidirections for later usage.
        distances[(u, v)] = d
        distances[(v, u)] = d

# Step 5: Generate initial route and optimize
# Make a shallow copy of the cities list to use as starting tour
initial_route = cities[:]
# Randomly permute the initial route to start the local search from a random solution.
random.shuffle(initial_route)
# Record current time before optimization
start_opt = time.time()
# Run the 2-opt local improvement
optimized_route, cost_history = two_opt(initial_route)  # cost_history presumably records intermediate costs
# Record end time after optimization
end_opt = time.time()
# Compute final cost after optimization
optimized_cost = cost(optimized_route)
# Compute the runtime
opt_runtime = end_opt - start_opt

# Step 6: Print results
print("\n--- Performance Metrics ---")
print("2-opt route:", optimized_route)
print(f"2-opt cost: {optimized_cost:.1f}")
print(f"2-opt run-time: {opt_runtime:.6f} seconds")

# Step 7: Plot optimized route
def plot_route(route, title):
    route = route + [route[0]]
    x = [city_coords[city][0] for city in route]
    y = [city_coords[city][1] for city in route]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], city, fontsize=12, color='red')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_route(optimized_route, "Optimized Route (2-opt)")

# Step 8: Visualize full graph with edge weights
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

pos = {city: city_coords[city] for city in cities}
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
labels = {edge: f"{weight:.1f}" for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.title("City Graph with Distances")
plt.show()

# Since the starting point is random, the result may be different.

========================================================================================================================
Christofides algorithm with 8 points

import math
import time
import networkx as nx
import itertools
import matplotlib.pyplot as plt

# Step 1: Cost function
def cost(route):
    total = 0
    for i in range(len(route) - 1):
        total += distances[(route[i], route[i+1])]
    return total

# Step 2: Christofides TSP algorithm
def christofides_tsp(graph):
    # Compute the mst graph to connect all nodes with the minimum total edge weight without forming cycles.
    mst = nx.minimum_spanning_tree(graph)
    # Find all vertices that have odd degrees
    odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]
    # Create a subgraph consisting only of these odd-degree vertices
    subgraph = graph.subgraph(odd_degree_nodes)
    # Find a Minimum Weight Perfect Matching on the subgraph
    # This pairs up all odd-degree nodes with the minimum total added costs
    matching = nx.algorithms.matching.min_weight_matching(subgraph)
    # Combine the MST and the matching edges to form an Eulerian multigraph
    eulerian_graph = nx.MultiGraph(mst)
    for u, v in matching:
        weight = graph[u][v]['weight']
 	      # Add each matched edge with its corresponding weight
        eulerian_graph.add_edge(u, v, weight=weight)
    # Check if the combined graph is Eulerian (every node has even degree)
    if not nx.is_eulerian(eulerian_graph):
        raise ValueError("Combined graph is not Eulerian")
    # Find an Eulerian circuit, i.e., a path that visits every edge exactly once and returns to the start.
    eulerian_circuit = list(nx.eulerian_circuit(eulerian_graph))
    # Create a list and set()
    tsp_tour = []
    visited = set()
    # Check everything in the circuit
    for u, v in eulerian_circuit:
        if u not in visited:
	       # Add into the list if not visit
            tsp_tour.append(u)
            visited.add(u)
    # Add the starting node to close the tour
    tsp_tour.append(tsp_tour[0])
    return tsp_tour

# Step 3: Define city coordinates
city_coords = {
    'A': (0, 0),
    'B': (2, 3),
    'C': (5, 4),
    'D': (6, 1),
    'E': (3, 8),
    'F': (1, 7),
    'G': (9, 5),
    'H': (1, 1)
}

# Step 4: Compute pairwise distances
# Create a new dictionary to keep the distances between cities
distances = {}
# Get a cities list of city_coords keys created
cities = list(city_coords.keys())
# Outer loop to ensure all cities are computed
for i in range(len(cities)):
    # Inner loop, ensuring that each unordered pair is computed once
    for j in range(i + 1, len(cities)):
	      # Assign variable for each pair
        u, v = cities[i], cities[j]
	      # Get the coordinated of u and v
        x1, y1 = city_coords[u]
        x2, y2 = city_coords[v]
	      # Get the distance between u and v
        d = math.hypot(x1 - x2, y1 - y2)
	      # Store them in bidirections for later usage.
        distances[(u, v)] = d
        distances[(v, u)] = d

# Step 5: Build graph
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

# Step 6: Christofides route and metrics
start_christofides = time.time()
christofides_route = christofides_tsp(graph)
end_christofides = time.time()
christofides_cost = cost(christofides_route)
christofides_time = end_christofides - start_christofides

# Step 7: Print metrics
print("\n--- Performance Metrics ---")
print(f"Christofides route: {christofides_route}")
print(f"Christofides cost: {christofides_cost:.2f}")
print(f"Christofides run-time: {christofides_time:.6f} seconds")

# Step 8: Plot optimized route
def plot_route(route, title):
    route = route + [route[0]]
    x = [city_coords[city][0] for city in route]
    y = [city_coords[city][1] for city in route]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], city, fontsize=12, color='red')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_route(christofides_route, "Optimized Route (Christofides Algorithm)")

# Step 9: Visualize full graph with edge weights
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

pos = {city: city_coords[city] for city in cities}
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
labels = {edge: f"{weight:.1f}" for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.title("City Graph with Distances")
plt.show()

========================================================================================================================
Merged Algorithm with 8 points

import math
import time
import networkx as nx
import itertools
import matplotlib.pyplot as plt

# Step 1: Cost function
def cost(route):
    total = 0
    for i in range(len(route) - 1):
        total += distances[(route[i], route[i+1])]
    return total

# Step 2: Christofides TSP algorithm
def christofides_tsp(graph):
    # Compute the mst graph to connect all nodes with the minimum total edge weight without forming cycles.
    mst = nx.minimum_spanning_tree(graph)
    # Find all vertices that have odd degrees
    odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]
    # Create a subgraph consisting only of these odd-degree vertices
    subgraph = graph.subgraph(odd_degree_nodes)
    # Find a Minimum Weight Perfect Matching on the subgraph
    # This pairs up all odd-degree nodes with the minimum total added costs
    matching = nx.algorithms.matching.min_weight_matching(subgraph)
    # Combine the MST and the matching edges to form an Eulerian multigraph
    eulerian_graph = nx.MultiGraph(mst)
    for u, v in matching:
        weight = graph[u][v]['weight']
 	      # Add each matched edge with its corresponding weight
        eulerian_graph.add_edge(u, v, weight=weight)
    # Check if the combined graph is Eulerian (every node has even degree)
    if not nx.is_eulerian(eulerian_graph):
        raise ValueError("Combined graph is not Eulerian")
    # Find an Eulerian circuit, i.e., a path that visits every edge exactly once and returns to the start.
    eulerian_circuit = list(nx.eulerian_circuit(eulerian_graph))
    # Create a list and set()
    tsp_tour = []
    visited = set()
    # Check everything in the circuit
    for u, v in eulerian_circuit:
        if u not in visited:
	       # Add into the list if not visit
            tsp_tour.append(u)
            visited.add(u)
    # Add the starting node to close the tour
    tsp_tour.append(tsp_tour[0])
    return tsp_tour

# Step 3: Merge with two-opt algorithm
def two_opt(route):
    best = route
    improved = True
    cost_history = [cost(best)] # Added for performance curve
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                if cost(new_route) < cost(best):
                    best = new_route
                    improved = True
                    cost_history.append(cost(best)) # Added for performance curve
        route = best
    return best, cost_history

# Step 4: Define city coordinates
city_coords = {
    'A': (0, 0),
    'B': (2, 3),
    'C': (5, 4),
    'D': (6, 1),
    'E': (3, 8),
    'F': (1, 7),
    'G': (9, 5),
    'H': (1, 1)
}

# Step 5: Compute pairwise distances
# Create a new dictionary to keep the distances between cities
distances = {}
# Get a cities list of city_coords keys created
cities = list(city_coords.keys())
# Outer loop to ensure all cities are computed
for i in range(len(cities)):
    # Inner loop, ensuring that each unordered pair is computed once
    for j in range(i + 1, len(cities)):
	      # Assign variable for each pair
        u, v = cities[i], cities[j]
	      # Get the coordinated of u and v
        x1, y1 = city_coords[u]
        x2, y2 = city_coords[v]
	      # Get the distance between u and v
        d = math.hypot(x1 - x2, y1 - y2)
	      # Store them in bidirections for later usage.
        distances[(u, v)] = d
        distances[(v, u)] = d

# Step 6: Build graph
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

# Step 7: Christofides route and metrics
start_time = time.time()
christofides_route = christofides_tsp(graph)
optimized_route, cost_history = two_opt(christofides_route)
end_time = time.time()
merged_runtime = end_time - start_time
christofedes_cost = cost(christofides_route)
merged_cost = cost(optimized_route)
accuracy_ratio = christofides_cost / merged_cost

if christofides_cost > 0:
    percentage_improvement = ((christofides_cost - merged_cost) / christofides_cost) * 100
else:
    percentage_improvement = 0.0

# Step 8: Print metrics
print("\n--- Performance Metrics ---")
print(f"Merged route: {optimized_route}")
print(f"Merged cost: {merged_cost:.2f}")
print(f"Merged run-time: {merged_runtime:.6f} seconds")
print(f"Accuracy ratio (Christofides / Optimal): {accuracy_ratio:.3f}")
print(f"Percentage Improvement (Baseline to Final): {percentage_improvement:.2f}%\n")

# Step 9: Plot optimized route
def plot_route(route, title):
    route = route + [route[0]]
    x = [city_coords[city][0] for city in route]
    y = [city_coords[city][1] for city in route]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], city, fontsize=12, color='red')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_route(christofides_route, "Optimized Route (Merged Algorithm)")

# Step 10: Visualize full graph with edge weights
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

pos = {city: city_coords[city] for city in cities}
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
labels = {edge: f"{weight:.1f}" for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.title("City Graph with Distances")
plt.show()

# Step 11: Plot convergence curve
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history)), cost_history, marker='o')
plt.title("Merged Algorithm Performance Curve")
plt.xlabel("Iteration")
plt.ylabel("Route Cost")
plt.grid(True)
plt.show()

========================================================================================================================
2-opt algorithm with 25 points

import networkx as nx
import itertools
import matplotlib.pyplot as plt
import time
import math
import random

# Step 1: Cost function
def cost(route):
    total = 0
    for i in range(len(route) - 1):
        total += distances[(route[i], route[i+1])]
    return total

# Step 2: 2-opt function
def two_opt(route):
    best = route
    improved = True
    cost_history = [cost(best)] # Added for performance curve
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                if cost(new_route) < cost(best):
                    best = new_route
                    improved = True
                    cost_history.append(cost(best)) # Added for performance curve
        route = best
    return best, cost_history

# Step 3: Define city coordinates
city_coords = {
    'A1': (0, 0),'A2': (10, 8), 'A3': (7, 0), 'A4': (4, 0), 'A5': (2, 9),
    'B1': (2, 3),'B2': (6, 3), 'B3': (7, 3), 'B4': (2, 10), 'B5': (4, 3),
    'C1': (15, 4),'C2': (5, 4), 'C3': (1, 4), 'C4': (5, 14), 'C5': (9, 4),
    'D1': (6, 1), 'D2': (6, 11), 'D3': (10, 1), 'D4': (0, 1), 'D5': (5, 1),
    'E1': (3, 8), 'E2': (13, 8), 'E3': (1, 8), 'E4': (9, 8), 'E5': (5, 8)

}

# Step 4: Compute pairwise distances
# Create a new dictionary to keep the distances between cities
distances = {}
# Get a cities list of city_coords keys created
cities = list(city_coords.keys())
# Outer loop to ensure all cities are computed
for i in range(len(cities)):
    # Inner loop, ensuring that each unordered pair is computed once
    for j in range(i + 1, len(cities)):
	      # Assign variable for each pair
        u, v = cities[i], cities[j]
	      # Get the coordinated of u and v
        x1, y1 = city_coords[u]
        x2, y2 = city_coords[v]
	      # Get the distance between u and v
        d = math.hypot(x1 - x2, y1 - y2)
	      # Store them in bidirections for later usage.
        distances[(u, v)] = d
        distances[(v, u)] = d

# Step 5: Generate initial route and optimize
# Make a shallow copy of the cities list to use as starting tour
initial_route = cities[:]
# Randomly permute the initial route to start the local search from a random solution.
random.shuffle(initial_route)
# Record current time before optimization
start_opt = time.time()
# Run the 2-opt local improvement
optimized_route, cost_history = two_opt(initial_route)  # cost_history presumably records intermediate costs
# Record end time after optimization
end_opt = time.time()
# Compute final cost after optimization
optimized_cost = cost(optimized_route)
# Compute the runtime
opt_runtime = end_opt - start_opt

# Step 6: Print results
print("\n--- Performance Metrics ---")
print("2-opt route:", optimized_route)
print(f"2-opt cost: {optimized_cost:.1f}")
print(f"2-opt run-time: {opt_runtime:.6f} seconds")

# Step 7: Plot optimized route
def plot_route(route, title):
    route = route + [route[0]]
    x = [city_coords[city][0] for city in route]
    y = [city_coords[city][1] for city in route]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], city, fontsize=12, color='red')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_route(optimized_route, "Optimized Route (2-opt)")

# Step 8: Visualize full graph with edge weights
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

pos = {city: city_coords[city] for city in cities}
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
labels = {edge: f"{weight:.1f}" for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.title("City Graph with Distances")
plt.show()

# Since the starting point is random, the result may be different.

========================================================================================================================
Christofides algorithm with 25 points

import math
import time
import networkx as nx
import itertools
import matplotlib.pyplot as plt

# Step 1: Cost function
def cost(route):
    total = 0
    for i in range(len(route) - 1):
        total += distances[(route[i], route[i+1])]
    return total

# Step 2: Christofides TSP algorithm
def christofides_tsp(graph):
    # Compute the mst graph to connect all nodes with the minimum total edge weight without forming cycles.
    mst = nx.minimum_spanning_tree(graph)
    # Find all vertices that have odd degrees
    odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]
    # Create a subgraph consisting only of these odd-degree vertices
    subgraph = graph.subgraph(odd_degree_nodes)
    # Find a Minimum Weight Perfect Matching on the subgraph
    # This pairs up all odd-degree nodes with the minimum total added costs
    matching = nx.algorithms.matching.min_weight_matching(subgraph)
    # Combine the MST and the matching edges to form an Eulerian multigraph
    eulerian_graph = nx.MultiGraph(mst)
    for u, v in matching:
        weight = graph[u][v]['weight']
 	      # Add each matched edge with its corresponding weight
        eulerian_graph.add_edge(u, v, weight=weight)
    # Check if the combined graph is Eulerian (every node has even degree)
    if not nx.is_eulerian(eulerian_graph):
        raise ValueError("Combined graph is not Eulerian")
    # Find an Eulerian circuit, i.e., a path that visits every edge exactly once and returns to the start.
    eulerian_circuit = list(nx.eulerian_circuit(eulerian_graph))
    # Create a list and set()
    tsp_tour = []
    visited = set()
    # Check everything in the circuit
    for u, v in eulerian_circuit:
        if u not in visited:
	       # Add into the list if not visit
            tsp_tour.append(u)
            visited.add(u)
    # Add the starting node to close the tour
    tsp_tour.append(tsp_tour[0])
    return tsp_tour

# Step 3: Define city coordinates
city_coords = {
    'A1': (0, 0),'A2': (10, 8), 'A3': (7, 0), 'A4': (4, 0), 'A5': (2, 9),
    'B1': (2, 3),'B2': (6, 3), 'B3': (7, 3), 'B4': (2, 10), 'B5': (4, 3),
    'C1': (15, 4),'C2': (5, 4), 'C3': (1, 4), 'C4': (5, 14), 'C5': (9, 4),
    'D1': (6, 1), 'D2': (6, 11), 'D3': (10, 1), 'D4': (0, 1), 'D5': (5, 1),
    'E1': (3, 8), 'E2': (13, 8), 'E3': (1, 8), 'E4': (9, 8), 'E5': (5, 8)

}

# Step 4: Compute pairwise distances
# Create a new dictionary to keep the distances between cities
distances = {}
# Get a cities list of city_coords keys created
cities = list(city_coords.keys())
# Outer loop to ensure all cities are computed
for i in range(len(cities)):
    # Inner loop, ensuring that each unordered pair is computed once
    for j in range(i + 1, len(cities)):
	      # Assign variable for each pair
        u, v = cities[i], cities[j]
	      # Get the coordinated of u and v
        x1, y1 = city_coords[u]
        x2, y2 = city_coords[v]
	      # Get the distance between u and v
        d = math.hypot(x1 - x2, y1 - y2)
	      # Store them in bidirections for later usage.
        distances[(u, v)] = d
        distances[(v, u)] = d

# Step 5: Build graph
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

# Step 6: Christofides route and metrics
start_christofides = time.time()
christofides_route = christofides_tsp(graph)
end_christofides = time.time()
christofides_cost = cost(christofides_route)
christofides_time = end_christofides - start_christofides

# Step 7: Print metrics
print("\n--- Performance Metrics ---")
print(f"Christofides route: {christofides_route}")
print(f"Christofides cost: {christofides_cost:.2f}")
print(f"Christofides run-time: {christofides_time:.6f} seconds")

# Step 8: Plot optimized route
def plot_route(route, title):
    route = route + [route[0]]
    x = [city_coords[city][0] for city in route]
    y = [city_coords[city][1] for city in route]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], city, fontsize=12, color='red')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_route(christofides_route, "Optimized Route (Christofides Algorithm)")

# Step 9: Visualize full graph with edge weights
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

pos = {city: city_coords[city] for city in cities}
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
labels = {edge: f"{weight:.1f}" for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.title("City Graph with Distances")
plt.show()

========================================================================================================================
Merged algorithm with 25 points

import math
import time
import networkx as nx
import itertools
import matplotlib.pyplot as plt

# Step 1: Cost function
def cost(route):
    total = 0
    for i in range(len(route) - 1):
        total += distances[(route[i], route[i+1])]
    return total

# Step 2: Christofides TSP algorithm
def christofides_tsp(graph):
    # Compute the mst graph to connect all nodes with the minimum total edge weight without forming cycles.
    mst = nx.minimum_spanning_tree(graph)
    # Find all vertices that have odd degrees
    odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]
    # Create a subgraph consisting only of these odd-degree vertices
    subgraph = graph.subgraph(odd_degree_nodes)
    # Find a Minimum Weight Perfect Matching on the subgraph
    # This pairs up all odd-degree nodes with the minimum total added costs
    matching = nx.algorithms.matching.min_weight_matching(subgraph)
    # Combine the MST and the matching edges to form an Eulerian multigraph
    eulerian_graph = nx.MultiGraph(mst)
    for u, v in matching:
        weight = graph[u][v]['weight']
 	      # Add each matched edge with its corresponding weight
        eulerian_graph.add_edge(u, v, weight=weight)
    # Check if the combined graph is Eulerian (every node has even degree)
    if not nx.is_eulerian(eulerian_graph):
        raise ValueError("Combined graph is not Eulerian")
    # Find an Eulerian circuit, i.e., a path that visits every edge exactly once and returns to the start.
    eulerian_circuit = list(nx.eulerian_circuit(eulerian_graph))
    # Create a list and set()
    tsp_tour = []
    visited = set()
    # Check everything in the circuit
    for u, v in eulerian_circuit:
        if u not in visited:
	       # Add into the list if not visit
            tsp_tour.append(u)
            visited.add(u)
    # Add the starting node to close the tour
    tsp_tour.append(tsp_tour[0])
    return tsp_tour

# Step 3: Merge with two-opt algorithm
def two_opt(route):
    best = route
    improved = True
    cost_history = [cost(best)] # Added for performance curve
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                if cost(new_route) < cost(best):
                    best = new_route
                    improved = True
                    cost_history.append(cost(best)) # Added for performance curve
        route = best
    return best, cost_history

# Step 4: Define city coordinates
city_coords = {
    'A1': (0, 0),'A2': (10, 8), 'A3': (7, 0), 'A4': (4, 0), 'A5': (2, 9),
    'B1': (2, 3),'B2': (6, 3), 'B3': (7, 3), 'B4': (2, 10), 'B5': (4, 3),
    'C1': (15, 4),'C2': (5, 4), 'C3': (1, 4), 'C4': (5, 14), 'C5': (9, 4),
    'D1': (6, 1), 'D2': (6, 11), 'D3': (10, 1), 'D4': (0, 1), 'D5': (5, 1),
    'E1': (3, 8), 'E2': (13, 8), 'E3': (1, 8), 'E4': (9, 8), 'E5': (5, 8)
}

# Step 5: Compute pairwise distances
# Create a new dictionary to keep the distances between cities
distances = {}
# Get a cities list of city_coords keys created
cities = list(city_coords.keys())
# Outer loop to ensure all cities are computed
for i in range(len(cities)):
    # Inner loop, ensuring that each unordered pair is computed once
    for j in range(i + 1, len(cities)):
	      # Assign variable for each pair
        u, v = cities[i], cities[j]
	      # Get the coordinated of u and v
        x1, y1 = city_coords[u]
        x2, y2 = city_coords[v]
	      # Get the distance between u and v
        d = math.hypot(x1 - x2, y1 - y2)
	      # Store them in bidirections for later usage.
        distances[(u, v)] = d
        distances[(v, u)] = d

# Step 6: Build graph
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

# Step 7: Christofides route and metrics
start_time = time.time()
christofides_route = christofides_tsp(graph)
optimized_route, cost_history = two_opt(christofides_route)
end_time = time.time()
merged_runtime = end_time - start_time
christofedes_cost = cost(christofides_route)
merged_cost = cost(optimized_route)
accuracy_ratio = christofides_cost / merged_cost

if christofides_cost > 0:
    percentage_improvement = ((christofides_cost - merged_cost) / christofides_cost) * 100
else:
    percentage_improvement = 0.0

# Step 8: Print metrics
print("\n--- Performance Metrics ---")
print(f"Merged route: {optimized_route}")
print(f"Merged cost: {merged_cost:.2f}")
print(f"Merged run-time: {merged_runtime:.6f} seconds")
print(f"Accuracy ratio (Christofides / Optimal): {accuracy_ratio:.3f}")
print(f"Percentage Improvement (Baseline to Final): {percentage_improvement:.2f}%\n")

# Step 9: Plot optimized route
def plot_route(route, title):
    route = route + [route[0]]
    x = [city_coords[city][0] for city in route]
    y = [city_coords[city][1] for city in route]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'bo-')
    for i, city in enumerate(route):
        plt.text(x[i], y[i], city, fontsize=12, color='red')
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_route(christofides_route, "Optimized Route (Merged Algorithm)")

# Step 10: Visualize full graph with edge weights
graph = nx.Graph()
for (u, v), w in distances.items():
    graph.add_edge(u, v, weight=w)

pos = {city: city_coords[city] for city in cities}
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
labels = {edge: f"{weight:.1f}" for edge, weight in nx.get_edge_attributes(graph, 'weight').items()}
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.title("City Graph with Distances")
plt.show()

# Step 11: Plot convergence curve
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history)), cost_history, marker='o')
plt.title("Merged Algorithm Performance Curve")
plt.xlabel("Iteration")
plt.ylabel("Route Cost")
plt.grid(True)
plt.show()

Done by Avis Oh Xin Wan UOW ID: 8465678
