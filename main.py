# #---------------------------------------------GITHUB SPECTRAL ALGO
# """
# DM1
#
# Implementation of the Spectral Partition Algorithm
# """
import itertools
import random

import numpy as np
import subprocess
import logging
import networkx as nx
import matplotlib.pyplot as plt
import time

# Function to import nodes for plotting
def import_nodes1(nodes_file):
    """
    Import the nodes from the file and prepare data for plotting
    """

    edges = []
    number_nodes = 0

    # Open the file
    with open(nodes_file, "r") as datafile:
        for line in datafile:
            node1, node2 = map(int, line.split())
            edges.append((node1, node2))
            number_nodes = max(number_nodes, node1, node2)

    number_nodes += 1  # Increment by one to account for zero-indexing

    return edges, number_nodes

# Function to plot original and partitioned graphs
def graph_plot(original_edges, partitioned_edges, num_nodes_original, subsets):
    # Create graphs for original and partitioned edges
    G_original = nx.Graph()
    G_partitioned = nx.Graph()

    # Add original edges to G_original
    G_original.add_edges_from(original_edges)

    # Add partitioned edges to G_partitioned
    G_partitioned.add_edges_from(original_edges)  # Retain original edges for the partitioned graph

    # Get positions for nodes using spring layout
    pos_original = nx.spring_layout(G_original)
    pos_partitioned = nx.spring_layout(G_partitioned)

    # Plot the graphs side by side
    plt.figure(figsize=(12, 5))

    # Plot the original graph
    plt.subplot(1, 2, 1)
    plt.title('Original Graph')
    nx.draw(G_original, pos_original, with_labels=False, node_size=20, node_color='skyblue', font_weight='bold',
            font_color='black')
    nx.draw_networkx_edges(G_original, pos_original, edgelist=G_original.edges(), width=0.5, alpha=0.25)
                           # Plot the partitioned graph with different subsets having different colors
    plt.subplot(1, 2, 2)
    plt.title('Partitioned Graph')
    for i, subset in enumerate(subsets):
        color = plt.cm.Set3(i / len(subsets))  # Get distinct colors for each subset
        nodes = [node for node in range(num_nodes_original) if node in subset]
        nx.draw_networkx_nodes(G_partitioned, pos_partitioned, nodelist=nodes, node_color=[color], node_size=20, label=f"Subset {i + 1}")

    nx.draw_networkx_edges(G_partitioned, pos_partitioned, edgelist=G_partitioned.edges(), width=0.5, alpha=0.25)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Function to import nodes for the spectral algorithm
def import_nodes(nodes_file):
    """
    Import the nodes from the file
    """

    edges = []
    adjacency_matrix = {}
    nodes_set = set()

    # Open the file
    with open(nodes_file, "r") as datafile:
        for line in datafile:
            node1, node2 = map(int, line.split())
            edges.append((node1, node2))
            nodes_set.update([node1, node2])

    number_nodes = len(nodes_set)

    nodes_array = list(nodes_set)#!!!!!!!!!!!!!

    logging.info("Imported {} nodes with {} edges from {}".format(number_nodes, len(edges), nodes_file))

    # Map nodes to their respective index
    node_index_map = {node: index for index, node in enumerate(nodes_set)}

    node_index_values = list(node_index_map.values())#!!!!!!!!!!!!!

    # Initialize empty adjacency matrix
    adjacency_matrix_array = np.zeros((number_nodes, number_nodes))

    # Fill the adjacency matrix based on the mentioned nodes
    for node1, node2 in edges:
        index1, index2 = node_index_map[node1], node_index_map[node2]
        adjacency_matrix_array[index1][index2] = 1
        adjacency_matrix_array[index2][index1] = 1

    important_sign = True
    for i in range(number_nodes):
        if nodes_array[i] != node_index_values[i]:
            important_sign = False


    return number_nodes, edges, adjacency_matrix_array, node_index_values, nodes_array, important_sign


# Function to compute the degree of each node
def degree_nodes(adjacency_matrix, number_nodes):
    """
    Compute the degree of each node
    Returns the vector of degrees
    """

    d = []
    for i in range(number_nodes):
        d.append(sum([adjacency_matrix[i][j] for j in range(number_nodes)]))

    return d

# Function to print the graph in a .gv file for visualization
def print_graph(number_nodes, edges, partition, outputfile):
    """
    Writes a .gv file to use with dot
    """
    with open("graph.gv", "w") as gv:
        gv.write("strict graph communities {")

        for node, community in enumerate(partition):
            gv.write("node{} [color={}];".format(node, "red" if community else "blue"))

        for node1, node2 in edges:
            gv.write("node{} -- node{};".format(node1, node2))

        gv.write("}")
        gv.close()

    subprocess.call(["dot", "-Tpng", "graph.gv", "-o", outputfile])

# Function to perform spectral partitioning on subsets
def algorithm_for_subsets(subset_array,edge_array):
    number_nodes = len(subset_array)
    edges = list(edge_array)

    # Initiate an empty set to hold unique nodes
    nodes_set = set()
    temp_starting_subset_array = min(subset_array)

    if temp_starting_subset_array == 0:
        temp_starting_subset_array = 1

    # Collect unique nodes from edges
    for edge in edges:
        node1, node2 = tuple(edge)  # Convert the set to a tuple of nodes
        nodes_set.update([node1, node2])

    number_nodes = len(nodes_set)

    nodes_array = list(nodes_set)

    # Map nodes to their respective index
    node_index_map = {node: index for index, node in enumerate(nodes_set)}
    node_index_values = list(node_index_map.values())

    # Check for the different values of those indices
    important_sign = True
    for i in range(number_nodes):
        if nodes_array[i] != node_index_values[i]:
            important_sign = False

    # Adjust node indices to avoid zero
    for i in range(len(node_index_values)):
        if node_index_values[i] == 0:
            node_index_values[i] = node_index_values[i]
        else:
            node_index_values[i] += temp_starting_subset_array


    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((number_nodes, number_nodes))

    # Populate adjacency matrix based on edge information
    for node1, node2 in edges:
        index1, index2 = node_index_map[node1], node_index_map[node2]
        adjacency_matrix[index1][index2] = 1
        adjacency_matrix[index2][index1] = 1

    # Compute node degrees and log the information
    degrees = degree_nodes(adjacency_matrix, number_nodes)
    logging.debug("Degrees: ", degrees)

    # Compute the Laplacian matrix
    laplacian_matrix = np.diag(degrees) - adjacency_matrix
    logging.debug("Laplacian matrix:\n", laplacian_matrix)

    # Perform eigen-decomposition to get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    logging.info("Found eigenvalues: ", eigenvalues)

    # Index of the second eigenvalue
    index_fnzev = np.argsort(eigenvalues)[1]
    logging.debug("Eigenvector for #{} eigenvalue ({}): ".format(
        index_fnzev, eigenvalues[index_fnzev]), eigenvectors[:, index_fnzev])

    # Partition on the sign of the eigenvector's coordinates
    partition = [val >= 0 for val in eigenvectors[:, index_fnzev]]

    # Compute the nodes in each partition
    nodes_in_partitions = []

    for i in range(2):
        nodes_in_partition = [node for (node, community) in enumerate(partition) if community == i]
        nodes_in_partitions.append(nodes_in_partition)

    # Adjust the partition if indices are not in order
    if important_sign == False:
        flattened_subsets = sum(nodes_in_partitions, [])
        value_to_replace = []
        replacement_value = []
        for i in range(number_nodes):
            if nodes_array[i] != flattened_subsets[i]:
                value_to_replace.append(flattened_subsets[i])
                replacement_value.append(nodes_array[i])
                flattened_subsets[i] = nodes_array[i]

        # Iterate through each sublist
        for sublist in nodes_in_partitions:
            for i in range(len(sublist)):
                if sublist[i] in value_to_replace:
                    sublist[i] = replacement_value[value_to_replace.index(sublist[i])]

    nodes_in_A = []
    nodes_in_B  = []

    for i, sublist in enumerate(nodes_in_partitions):
        if i == 0:
            nodes_in_A.extend(sublist)
        elif i == 1:
            nodes_in_B.extend(sublist)

    edges_in_between = []

    for edge in edges:
        node1, node2 = edge
        if node1 in nodes_in_A and node2 in nodes_in_B \
                or node1 in nodes_in_B and node2 in nodes_in_A:
            edges_in_between.append(edge)

    return nodes_in_A, nodes_in_B, number_nodes, edges, edges_in_between, partition



# Main spectral partitioning algorithm
def algorithm(nodes_file, k):
    """
    Spectral Partitioning Algorithm for k subsets
    """
    # Import nodes from the provided file and assign necessary values
    number_nodes, edges, adjacency_matrix, node_index_values, nodes_array, important_sign = import_nodes(nodes_file)
    logging.debug("Adjacency matrix:\n", adjacency_matrix)

    # Create a copy of 'edges' list named 'for_subset_edges'
    for_subset_edges = []
    for i in range(len(edges)):
        for_subset_edges.append(edges[i])

    # Compute degrees of nodes based on adjacency matrix
    degrees = degree_nodes(adjacency_matrix, number_nodes)
    logging.debug("Degrees: ", degrees)

    # Compute the Laplacian matrix using degree and adjacency matrices
    laplacian_matrix = np.diag(degrees) - adjacency_matrix  # n^3
    logging.debug("Laplacian matrix:\n", laplacian_matrix)

    # Compute eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix) # n^3
    logging.info("Found eigenvalues: ", eigenvalues)

    # Find the indices of the smallest k eigenvalues
    #x = k - 1  # to :k then cahnge the indexes[i] to indexes community will be like if not community with if clause !!!

    indexes = np.argsort(eigenvalues)[1]  #:k # the best choice is to take the eigenvalue second smallest[1]!!!

    for i in range(k):
        logging.debug(
            f"Eigenvector for the {i + 1}th smallest eigenvalue ({eigenvalues[indexes]}): {eigenvectors[:, indexes]}")

    # Partition based on the sign of the k eigenvectors' coordinates
    partition = [val >= 0 for val in eigenvectors[:, indexes]]

    subsets_for_printing = []
    edges_for_printing = []

    # Compute the nodes in each partition
    nodes_in_partitions = []
    for i in range(k):
        nodes_in_partition = [node for (node, community) in enumerate(partition) if community == i]
        nodes_in_partitions.append(nodes_in_partition)
        subsets_for_printing.append(nodes_in_partition)

    if k > 2:
        del subsets_for_printing[k-1]

    # Reassign nodes if indices are not in order
    if important_sign == False:
        flattened_subsets = sum(nodes_in_partitions, [])
        value_to_replace = []
        replacement_value = []
        for i in range(number_nodes):
            if nodes_array[i] != flattened_subsets[i]:
                value_to_replace.append(flattened_subsets[i])
                replacement_value.append(nodes_array[i])
                flattened_subsets[i] = nodes_array[i]

        # Iterate through each sublist
        for sublist in nodes_in_partitions:
            for i in range(len(sublist)):
                if sublist[i] in value_to_replace:
                    sublist[i] = replacement_value[value_to_replace.index(sublist[i])]

    edges_between_subsets = {}

    # Create a dictionary to hold edges between different subsets
    for i in range(k):
        for j in range(i + 1, k):
            edges_between_subsets[(i, j)] = []

    edges_between_subsets = {key: [] for key in itertools.combinations(range(k), 2)}

    # Compute edges between different subsets
    for edge in edges:
        node1, node2 = edge
        for subset_pair in itertools.combinations(range(k), 2):
            subset1, subset2 = subset_pair
            if (node1 in nodes_in_partitions[subset1] and node2 in nodes_in_partitions[subset2]) or \
                    (node1 in nodes_in_partitions[subset2] and node2 in nodes_in_partitions[subset1]):
                edges_between_subsets[subset_pair].append(edge)
                edges_for_printing.append(edge)

    if k > 2:
        #------------------------------------------------------------------------------- for k subsets
        #finding empty arrays
        empty_arrays = [subset for subset in nodes_in_partitions if len(subset) == 0]
        # Extracting subsets that are not empty
        subsets_array = nodes_in_partitions[:-len(empty_arrays)]
        subsets_edges = [[] for _ in range(len(subsets_array))]  # Initialize subsets_edges

        for i in range(len(subsets_array) - 1):
            for edge in for_subset_edges:
                node1, node2 = edge
                if (node1 in subsets_array[i] and node2 in subsets_array[i]) or \
                        (node1 in subsets_array[i + 1] and node2 in subsets_array[i + 1]):
                    if node1 in subsets_array[i] and node2 in subsets_array[i]:
                        subsets_edges[i].append(edge)
                    else:
                        subsets_edges[i+1].append(edge)

        nodes_in_A1 = []
        nodes_in_A2= []
        number_nodesA = []
        edgesA = []
        edges_in_betweenA = []
        len_edges_in_betweenA = []
        partitionA = []

        # NOW I HAVE subsets_array, subsets_edges
        # Perform an algorithm for subsets if there are more than two subsets
        for x in range(len(empty_arrays)):
            for i in range(len(subsets_array)):
                if len(subsets_array[i]) == 1:
                    nodes_in_A1.append([])
                    nodes_in_A2.append([])
                    number_nodesA.append(0)
                    edgesA.append([])
                    edges_in_betweenA.append([])
                    len_edges_in_betweenA.append(0)
                    partitionA.append([])
                    continue
                else:
                    # Call algorithm_for_subsets function to partition subsets
                    (temp_nodes_a1, temp_nodes_a2, temp_number_nodes, temp_edges,
                     temp_edges_in_between, temp_partition) = algorithm_for_subsets(subsets_array[i], subsets_edges[i])

                    if len(subsets_array[i]) != (len(temp_nodes_a1) + len(temp_nodes_a2)):
                        print("sadfa")
                        nodes_in_A1.append([])
                        nodes_in_A2.append([])
                        number_nodesA.append(0)
                        edgesA.append([])
                        edges_in_betweenA.append([])
                        len_edges_in_betweenA.append(30)
                        partitionA.append([])
                    else:
                        nodes_in_A1.append(temp_nodes_a1)
                        nodes_in_A2.append(temp_nodes_a2)
                        number_nodesA.append(temp_number_nodes)
                        edgesA.append(temp_edges)
                        edges_in_betweenA.append(temp_edges_in_between)
                        len_edges_in_betweenA.append(len(edges_in_betweenA[i]))
                        partitionA.append(temp_partition)

            # Finding subsets with non-zero edges in between
            non_zero_values = [value for value in len_edges_in_betweenA if value != 0]
            min_value = min(non_zero_values)

            min_indices = [i for i, value in enumerate(len_edges_in_betweenA) if value == min_value]

            if len(min_indices) > 1:
                max_nodes = 0
                selected_index = None
                for idx in min_indices:
                    num_nodes = number_nodesA[idx]
                    if num_nodes > max_nodes:
                        max_nodes = num_nodes
                        selected_index = idx
                min_index = selected_index
            else:
                min_index = len_edges_in_betweenA.index(min_value)

            # Adjust subsets based on the computed values


            del subsets_array[min_index]

            temp_subsets_array = []
            for j in range(len(subsets_array)):
                temp_subsets_array.append(subsets_array[j])

            subsets_array.clear()  # Clear subsets_edges to make it an empty list
            subsets_array.append(nodes_in_A1[min_index])
            subsets_array.append(nodes_in_A2[min_index])

            for j in range(len(temp_subsets_array)):
                subsets_array.append(temp_subsets_array[j])

            combine_subsets = [nodes_in_A1[min_index], nodes_in_A2[min_index]]
            combine_subsets_edges = [[] for _ in range(len(combine_subsets))]
            for j in range(len(combine_subsets) - 1):
                for edge in edgesA[min_index]:
                    node1, node2 = edge
                    if (node1 in combine_subsets[j] and node2 in combine_subsets[j]) or \
                            (node1 in combine_subsets[j + 1] and node2 in combine_subsets[j + 1]):
                        if node1 in combine_subsets[j] and node2 in combine_subsets[j]:
                            combine_subsets_edges[j].append(edge)
                        else:
                            combine_subsets_edges[j + 1].append(edge)

            del subsets_edges[min_index]

            temp_subsets_edges = []
            for j in range(len(subsets_edges)):
                temp_subsets_edges.append(subsets_edges[j])

            subsets_edges.clear()  # Clear subsets_edges to make it an empty list

            for j in range(len(combine_subsets_edges)):
                subsets_edges.append(combine_subsets_edges[j])

            for j in range(len(temp_subsets_edges)):
                subsets_edges.append(temp_subsets_edges[j])

            # Ensure subsets_for_printing has enough elements
            if len(subsets_array) > len(subsets_for_printing):
                subsets_for_printing.extend([None] * (len(subsets_array) - len(subsets_for_printing)))

            # Assign values from subsets_array to subsets_for_printing
            for j in range(len(subsets_array)):
                subsets_for_printing[j] = subsets_array[j]

            edges_for_printing.extend(tuple(edges_in_betweenA[min_index]))

            nodes_in_A1.clear()  # Clear subsets_edges to make it an empty list
            nodes_in_A2.clear()  # Clear subsets_edges to make it an empty list
            number_nodesA.clear()  # Clear subsets_edges to make it an empty list
            edgesA.clear()
            #edgesA = subsets_edges.copy()  # Creates a shallow copy of subsets_edges
            edges_in_betweenA.clear()  # Clear subsets_edges to make it an empty list
            len_edges_in_betweenA.clear()
            partitionA.clear()  # Clear subsets_edges to make it an empty list


        return subsets_array, subsets_edges, subsets_for_printing, edges_for_printing
    else:
        return nodes_in_partitions, edges_between_subsets, subsets_for_printing, edges_for_printing


# Main function that handles command line arguments and execution
if __name__ == '__main__':
    import argparse

    # #----------------------------
    # num_nodes = 50
    # num_edges = 80
    #
    # edges = set()
    #
    # while len(edges) < num_edges:
    #     node1 = random.randint(0, num_nodes - 1)
    #     node2 = random.randint(0, num_nodes - 1)
    #     if node1 != node2:
    #         edge = (min(node1, node2), max(node1, node2))
    #         edges.add(edge)
    #
    # # Display the edges in the required format
    # for edge in edges:
    #     print(f"{edge[0]} {edge[1]}")
    # #----------------------------

    # Configure logging
    FORMAT = '%(asctime)s.%(msecs)03d %(message)s'
    logging.basicConfig(format=FORMAT, datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser(description="Compute the partition of a "
        "graph using the Spectral Partition Algorithm.")

    parser.add_argument('--nodes-file', '-f', help='the file containing the nodes',
                        default='demo_nodes.txt')
    parser.add_argument('--output-file', '-o', help='the filename of the'
                        ' communities PNG graph to be written')

    args = parser.parse_args()

    #Run the algorithm
    start_time = time.time()

    subsets, edges_partitioned, subsets_for_printing, edges_for_printing = algorithm(args.nodes_file, 2)

    end_time = time.time()
    execution_time = end_time - start_time
    print("execution_time: ", execution_time)

    for i in range(len(subsets_for_printing)):
        print(f"Subset {i + 1}: ", subsets_for_printing[i])

    for i in range(len(subsets_for_printing)):
        print(f"Number of Subset {i + 1}: ", len(subsets_for_printing[i]))

    # Dictionary to hold edges between subsets
    subset_edges = {}

    # Iterate through each subset pair
    for i, subset1 in enumerate(subsets_for_printing):
        for j, subset2 in enumerate(subsets_for_printing[i + 1:], start=i):
            # Initialize edge count between subset pairs
            subset_edges[(i + 1, j + 2)] = 0
            # Iterate through each edge
            for edge in edges_for_printing:
                node1, node2 = edge
                # Check if nodes are in the subsets
                if node1 in subset1 and node2 in subset2 or node1 in subset2 and node2 in subset1:
                    subset_edges[(i + 1, j + 2)] += 1  # Increment edge count

    # Print edges between subsets
    for subset_pair, count in subset_edges.items():
        if count != 0:
            print(f"Edges between Subset {subset_pair[0]} and Subset {subset_pair[1]}: {count} edges")

    edges_partitioned = [edge for sublist in edges_partitioned for edge in sublist]
    edges_original, num_nodes_original = import_nodes1(args.nodes_file)
    graph_plot(edges_original, edges_partitioned, num_nodes_original, subsets_for_printing)