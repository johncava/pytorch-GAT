
# Returns an array that has an array of neighbors for each node including itself
def get_neighbors(graph):
    neighbors = []
    for node in graph:
        node_neighbors = []
        for index, val in enumerate(node):
            if val == 1:
                node_neighbors.append(index)
        neighbors.append(node_neighbors)
    return neighbors