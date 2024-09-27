#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import TypedDict
import pickle

import networkx
import pandas
from modules.applications.Application import *
from modules.applications.optimization.Optimization import Optimization
import numpy as np
from utils import start_time_measurement, end_time_measurement


class MKC(Optimization):
    """
    In planning problems, there will be tasks to be done, and some of them may be mutually exclusive.
    We can translate this into a graph where the nodes are the tasks and the edges are the mutual exclusions.
    The maximum independent set (MIS) problem is a combinatorial optimization problem that seeks to find the largest
    subset of vertices in a graph such that no two vertices are adjacent.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("MKC")
        self.submodule_options = ["QrispBacktrackingColoring"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
        ]

    def get_solution_quality_unit(self) -> str:
        return "Set size"

    def get_default_submodule(self, option: str) -> Core:
        if option == "QrispBacktrackingColoring":
            from modules.applications.optimization.MKC.mappings.QrispBacktrackingColoring import MaxKappaGraphColoring  # pylint: disable=C0415
            return MaxKappaGraphColoring()
        else:
            raise NotImplementedError(f"Mapping Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this application.
        This is empty in the case of backtracking (for now)

        """
        return {
                        "problemType": {  # number of optimization iterations
                "values": ["Graph", "Sudoku"],
                "description": "Which problem do you want to generate?"
            }
 
        }

    class Config(TypedDict):
        """
        Attributes of a valid config
        Empty for now


        """


    def generate_problem(self, config: Config) -> networkx.Graph:
        """
        Generates a graph to solve the MIS for.

        :param config: Config specifying the size and connectivity for the problem
        :type config: Config
        :return: networkx graph representing the problem
        :rtype: networkx.Graph
        """

        if config is None:
            config = {"size": 3,
                      "spacing": 1,
                      "filling_fraction": 0.5}

        # check if config has the necessary information

        problemType = config["problemType"]

        # create problem TBD
        if problemType == "Sudoku":
        #graph = pandas.read_csv(r'src\modules\applications\optimization\Backtracking\data\sudoku_problem.csv', sep = ";")
            from numpy import genfromtxt
            graph  = genfromtxt(r'src\modules\applications\optimization\MKC\data\sudoku_problem.csv', delimiter=';')
            #graph.to_numpy()
        else:
            from src.modules.applications.optimization.MKC.data.graph_coloring import problem_in
            graph   = problem_in
            num_empty_fields = sum([1 for index in range(len(problem_in[0])) if problem_in[0][index] == -1])
        logging.info("Created Sudoku problem from sudoku_problem.csv, with the following attributes:")
        #for item in range(graph.shape[0]):
        logging.info(print(graph))

        self.application = graph
        return graph.copy()

    def process_solution(self, solution: list) -> (list, float):
        """
        Returns list of visited nodes and the time it took to process the solution

        :param solution: Unprocessed solution
        :type solution: list
        :return: Processed solution and the time it took to process it
        :rtype: tuple(list, float)
        """
        start_time = start_time_measurement()

        return solution, end_time_measurement(start_time)

    def validate(self, solution: list) -> (bool, float):
        """
        Checks if the solution is an independent set

        :param solution: List containing the nodes of the solution
        :type solution: list
        :return: Boolean whether the solution is valid and time it took to validate
        :rtype: tuple(bool, float)
        """
        start = start_time_measurement()
        is_valid = True

        sudoku_board = self.application

           # Receives a quantum array with the values for the empty fields and
        # returns a QuantumBool, that is True if the Sudoku solution is valid
        
        from qrisp import QuantumBool, QuantumVariable, mcx, auto_uncompute, QuantumArray, QuantumFloat

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
        def check_sudoku_board(empty_field_values : QuantumArray):
            
            # Create a quantum array, that contains a mix of the classical and quantum values
            shape = sudoku_board.shape
            quantum_sudoku_board = np.zeros(shape = sudoku_board.shape, dtype = "object")        
            
            quantum_value_list = list(empty_field_values)
            
            # Fill the board
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if sudoku_board[i,j] == -1:
                        quantum_sudoku_board[i, j] = quantum_value_list.pop(0)
                    else:
                        quantum_sudoku_board[i, j] = int(sudoku_board[i, j])
            
            # Go through the conditions that need to be satisfied
            check_values = []   
            for i in range(shape[0]):
                
                # Rows
                check_values.append(element_distinctness(quantum_sudoku_board[i,:]))
                
                # Columns
                check_values.append(element_distinctness(quantum_sudoku_board[j,:]))
            
            # Squares
            # this one needs to be changed
            top_left_square = quantum_sudoku_board[:2,:2].flatten()
            top_right_square = quantum_sudoku_board[2:,:2].flatten()
            bot_left_square = quantum_sudoku_board[:2,2:].flatten()
            bot_right_square = quantum_sudoku_board[2:,2:].flatten()
            
            check_values.append(element_distinctness(top_left_square))
            check_values.append(element_distinctness(top_right_square))
            check_values.append(element_distinctness(bot_left_square))
            check_values.append(element_distinctness(bot_right_square))
            
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
        
        @auto_uncompute
        def check_graph_coloring(empty_field_values : QuantumArray):
            #print(graph)
            # Create a quantum array, that contains a mix of the classical and quantum values
            graph = self.application
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
        
        empty_field_values = QuantumArray(qtype = QuantumFloat(2), shape = (len(solution)))
        empty_field_values[:] = solution
        if isinstance(self.application, (np.ndarray, np.generic)):
        # problemType = "sudoku"
            try: 
                test = check_sudoku_board(empty_field_values)
                is_valid =  test.get_measurement() == {True : 1.0}
            except: 
                pass

        elif isinstance(self.application, list):
        # problemType = "sudoku"
            try: 
                test = check_graph_coloring(empty_field_values)
                is_valid =  test.get_measurement() == {True : 1.0}
            except: 
                pass

        return is_valid, end_time_measurement(start)

    def evaluate(self, solution: list) -> (int, float):
        """
        Calculates the size of the solution

        :param solution: List containing the nodes of the solution
        :type solution: list
        :return: Set size, time it took to calculate the set size
        :rtype: tuple(int, float)
        """
        start = start_time_measurement()
        set_size = len(solution)

        logging.info(f"Size of solution: {set_size}")

        return set_size, end_time_measurement(start)

    def save(self, path: str, iter_count: int) -> None:
        with open(f"{path}/graph_iter_{iter_count}.gpickle", "wb") as file:
            pickle.dump(self.application, file, pickle.HIGHEST_PROTOCOL)
