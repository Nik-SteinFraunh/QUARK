import networkx as nx 
import matplotlib.pyplot as plt
import numpy as np

input_obj = [1, -1, 3, 1, -1 , 2]
input_obj2 = [(0,2),(0,1),(3,5),(2,4),(1,4)]

inputa = [input_obj, input_obj2]

def input_obj_transformationa(input_obj):
    """
    Convert a 4x4 Sudoku problem into a graph coloring problem using networkx.

    Parameters:
    - sudoku_board: 4x4 numpy array with numbers 0 to 3 for set fields and -1 for empty fields.

    Returns:
    - G: networkx graph representing the Sudoku problem.
    - empty_nodes: list of nodes corresponding to the empty fields.
    """
    if type(input_obj) == nx.graph:
        #TBD
        adj_matrix = nx.adjacency_matrix(input_obj)
        pass

    #Need to adjust to structure
    input_nodes = input_obj[0]
    input_edges = input_obj[1]

    edge_copy = input_edges.copy()
    adj_dict = {}
    for edge in edge_copy:
        for index in range(len(edge)):
            if edge[index] in list(adj_dict.keys()):
                adj_dict[edge[index]] +=[edge[1-index]]
            else:
                adj_dict.setdefault(edge[index], [edge[1-index]])
    #build a connection dict
    

    # Create an empty graph
    G = nx.Graph()
    empty_nodes = []
    # Add nodes and edges
    for i in range(len(input_nodes)):
        
        #test this for now, but might need check for double iteration over same edge
        if input_nodes[i] == -1:
            # Add node for each empty cell
            node = i
            adj_nodes = adj_dict[i]
            empty_nodes.append(node)
            G.add_node(node)
            for adj in adj_nodes:
                if input_nodes[adj] == -1: 
                    G.add_edge(node, adj, edge_type = "qq")
                else:
                    G.add_edge(node, adj, edge_type = "cq")


    return G, empty_nodes




def extract_coloring(input_obj):
    """
    Takes a Sudoku board in the form of a numpy array
    where the empty fields are indicated by the value -1.

    Returns two lists:
    1. The quantum-quantum comparisons in the form of a list[(int, int)]
    2. The batched classical-quantum comparisons in the form dict({int : list[int]})
    """
    #Need to adjust to structure
    input_nodes = input_obj[0]
    input_edges = input_obj[1]

    num_empty_fields = sum([1 for index in range(len(input_nodes)) if input_nodes[index] == -1])
    print(num_empty_fields)
    # Generate the comparison graph
    graph, empty_nodes = input_obj_transformationa( input_obj)

    # Generate the list of required comparisons

    # This dictionary contains the classical-quantum comparisons for each
    # quantum entry

    cq_checks = {q_assignment_index : [] for q_assignment_index in range(num_empty_fields)}
    #print(cq_checks)
    # This dictionary contains the quantum-quantum comparisons as tuples
    qq_checks = []

    # Each edge of the graph corresponds to a comparison.
    # We therefore iterate over the edges distinguish between the classical-quantum
    # and quantum-quantum comparisons

    for edge in graph.edges():
        edge_type = graph.get_edge_data(*edge)["edge_type"]

        # Append the quantum-quantum comparison to the corresponding list
        if edge_type == "qq":
            assigment_index_0 = empty_nodes.index(edge[0])
            assigment_index_1 = empty_nodes.index(edge[1])

            qq_checks.append((assigment_index_0, assigment_index_1))

        # Append the classical quantum comparison to the corresponding dictionary
        elif edge_type == "cq":

            if input_nodes[edge[1]] == -1:
                q_assignment_index = empty_nodes.index(edge[1])
                cq_checks[q_assignment_index].append(input_nodes[edge[0]])
            else:
                q_assignment_index = empty_nodes.index(edge[0])
                cq_checks[q_assignment_index].append(input_nodes[edge[1]])

    return qq_checks, cq_checks

""" Ga, empyt = input_obj_transformationa(inputa)

nx.draw(Ga, with_labels = True)
plt.show() """

checks1, checks2 = extract_coloring(inputa)

print(checks1)

print(checks2)
