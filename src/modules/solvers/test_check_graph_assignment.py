import numpy as np



from qrisp import QuantumBool, QuantumVariable, mcx, auto_uncompute, QuantumArray, QuantumFloat

#start = start_time_measurement()
is_valid = True

nodes_in = [2, 0, 1,-1 ,3,1,2, 2, -1, 0, -1 ]
edges_in = [(0,2),(0,1),(3,5),(2,4),(4,5),(1,5),(7,8),(7,10),(2,8),(3,9),(2,9),(2,10),(8,10),(6,10)]


qType = QuantumFloat(2)

graph = [nodes_in,edges_in]

sol = [2, 0, 3]
empty = QuantumArray(qType)
empty[:] = sol



# Receives a quantum array with the values for the empty fields and
# returns a QuantumBool, that is True if the Sudoku solution is valid



def element_distinctness(iterable):

    n = len(iterable)
    
    comparison_list = []
    
    for i in range(n):
        for j in range(i+1, n):
            
            # If both elements are classical and agree, return a QuantumBool with False
            if not isinstance(iterable[i], QuantumVariable) and not isinstance(iterable[j], QuantumVariable):
                if iterable[i] == iterable[j]:
                    res = QuantumBool()
                    res[:] = False
                    return res
                else:
                    continue
            
            # If atleast one of the elements is quantum, do a comparison
            comparison_list.append(iterable[i] != iterable[j])
    
    if len(comparison_list) == 0:
        return None
    
    res = QuantumBool()
    
    mcx(comparison_list, res)
    
    # Using recompute here reduces the qubit count dramatically 
    # More information here https://qrisp.eu/reference/Core/Uncomputation.html#recomputation
    for qbl in comparison_list: qbl.uncompute(recompute = False)
    
    return res

@auto_uncompute
def check_graph_coloring(empty_field_values : QuantumArray):
    #print(graph)
    # Create a quantum array, that contains a mix of the classical and quantum values
    nodes = graph[0]
    edges = graph[1]
    quantum_graph = nodes
    
    quantum_value_list = list(empty_field_values)
    
    # Fill the board
    for i in range(len(nodes)):
        
            if nodes[i] == -1:
                quantum_graph[i] = quantum_value_list.pop(0)
            else:
                continue
    
    # Go through the conditions that need to be satisfied
    check_values = []   

    for edge in edges:
        check_values.append(element_distinctness([quantum_graph[edge[0]], quantum_graph[edge[1]]]))
        
    #print(check_values)
    # element_distinctness returns None if only classical values have been compared
    # Filter these out
    i = 0
    while i < len(check_values):
        if check_values[i] is None:
            check_values.pop(i)
            continue
        i += 1
    
    # Compute the result
    res = QuantumBool()
    mcx(check_values, res)
    
    return res
    

test = check_graph_coloring(empty)
print(test)
is_valid =  test.get_measurement() == {True : 1.0}
print(is_valid)



