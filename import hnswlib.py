import hnswlib

# Define parameters for the HNSW index
dim = 128  # Dimensionality of your data vectors
M = 16  # Number of neighbors to connect to per node
ef_construction = 200  # Number of neighbors to explore during construction
ef_search = 100  # Number of neighbors to explore during search

# Initialize the HNSW index
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=10000, ef_construction=ef_construction, M=M)

# Example data (replace with your actual data)
data = [[0.1, 0.2, 0.3, ...],  # Vector 1
        [0.4, 0.5, 0.6, ...],  # Vector 2
        ...]  # More data points

# Add data to the index
index.add_items(data)
index.ef_search = ef_search  # Set search efficiency parameter

# Define a query vector
query = [0.7, 0.8, 0.9, ...]

# Perform nearest neighbor search
labels, distances = index.knn_query(query, k=5)

# Print the labels and distances of the 5 nearest neighbors
for label, distance in zip(labels[0], distances[0]):
  print(f"Neighbor label: {label}, Distance: {distance}")
