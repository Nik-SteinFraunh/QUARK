from qrisp import auto_uncompute, QuantumBool, QuantumDictionary, mcx, QuantumFloat, control
import numpy as np

import matplotlib.pyplot as plt


import networkx as nx

















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
import logging
from typing import Tuple
from typing import TypedDict

import os
import numpy as np

# from qiskit_ibm_runtime import QiskitRuntimeService

from modules.solvers.Solver import *
from utils import start_time_measurement, end_time_measurement


class QRISPQAOASolverMKC(Solver):
    """
    Qrisp QIRO.
    run the QIRO implementation within the qrisp simulator Backend. Further Backends TBD
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        # self.submodule_options = ["qasm_simulator", "qasm_simulator_gpu", "ibm_eagle"]
        self.submodule_options = ["qrisp_simulator"]

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Return requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "qrisp",
                "version": "0.49"
            }
        ]

    def get_default_submodule(self, option: str) -> Core:
        # TBD?
        if option == "qrisp_simulator":
            from modules.devices.HelperClass import HelperClass  # pylint: disable=C0415
            return HelperClass("qrisp_simulator")

        else:
            raise NotImplementedError(f"Device Option {option} not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver.

        :return:
                 .. code-block:: python

                              return {
                                        "shots": {  # number measurements to make on circuit
                                            "values": list(range(10, 500, 30)),
                                            "description": "How many shots do you need?"
                                        },
                                        "iterations": {  # number measurements to make on circuit
                                            "values": [1, 5, 10, 20, 50, 75],
                                            "description": "How many iterations do you need? Warning: When using\
                                            the IBM Eagle Device you should only choose a lower number of\
                                            iterations, since a high number would lead to a waiting time that\
                                            could take up to mulitple days!"
                                        },
                                        "depth": {
                                            "values": [2, 3, 4, 5, 10, 20],
                                            "description": "How many layers for QAOA (Parameter: p) do you want?"
                                        },
                                        "method": {
                                            "values": ["classic", "vqe", "qaoa"],
                                            "description": "Which Qiskit solver should be used?"
                                        },
                                        "optimizer": {
                                            "values": ["POWELL", "SPSA", "COBYLA"],
                                            "description": "Which Qiskit solver should be used? Warning: When\
                                            using the IBM Eagle Device you should not use the SPSA optimizer,\
                                            since it is not suited for only one evaluation!"
                                        }
                                    }

        """
        return {

            "backend": {  # number of optimization iterations
                "values": ["LocalSimulator", "IBM_MPSSimulator (TBD)"],
                "description": "Which Simulator do you want to use?"
            },
            "shots": {  # number measurements to make on circuit
                "values": [10, 500, 1000, 2000, 5000, 10000],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": int,
                "description": "How many shots do you need?"
            },
            "depth": {  # number measurements to make on circuit
                "values": [3,5,7],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": int,
                "description": "Which depth do you want for the QAOA circuit?"
            },
            "max_iter": {  # number measurements to make on circuit
                "values": [10, 20, 50, 100],
                "custom_input": True,
                "allow_ranges": True,
                "postproc": int,
                "description": "How many optimization iterations do you want?"
            }
        }

        ##############FURTHER OPTIONS TO BE INCLUDED
        """ 
            # do i want to do something here?
            "method": {
                "values": ["classic", "vqe", "qaoa"],
                "description": "Which Qiskit solver should be used?"
            },
            "optimizer": {
                "values": ["not", "yet", "implemented"],
                "description": "Which QIRO solver should be used?"
            } """

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            shots: int
            depth: int
            iterations: int
            layers: int
            method: str

        """
        problemType: str 
        shots: int
        depth: int
        iterations: int
        layers: int
        method: str



    def run(self, mapped_problem: any, device_wrapper: any, config: Config, **kwargs: dict):
        """
        Run Qrisp QIRO on the local Qrisp simulator

        :param mapped_problem: dictionary with the keys 'graph' and 't'
        :type mapped_problem: any
        :param device_wrapper: instance of device
        :type device_wrapper: any
        :param config:
        :type config: Config
        :param kwargs: no additionally settings needed
        :type kwargs: any
        :return: Solution, the time it took to compute it and optional additional information
        :rtype: tuple(list, float, dict)
        """
        
        shots = config["shots"]
        depth = config["depth"]
        backend = config["backend"]
        max_iter = config["max_iter"]

        import pandas
        from numpy import genfromtxt
        
        """ for row in range(df.shape[0]):
            row_list = []
            for col in range(df.shape[1]):
                row_list.append(df.iloc[row][col])
            sudoku_helper.append(row_list) """

        import networkx as nx
        start = start_time_measurement()


        from src.modules.applications.optimization.MKC.data.graph_coloring import problem_in
        problemToSolve   = problem_in
        num_empty_fields = sum([1 for index in range(len(problem_in[0])) if problem_in[0][index] == -1])
        eFI = {} # empty_Field_idnex
        iterat = 0
        for index in range(len(problem_in[0])):
            if problem_in[0][index] == -1:
                eFI.setdefault(index,iterat)
                iterat +=1

        from qrisp.algorithms.qaoa import QuantumArray, QuantumColor, QAOAProblem, create_coloring_cl_cost_function, apply_XY_mixer, p, control
        from operator import itemgetter
        import matplotlib.pyplot as plt
        import random

        
        full_nodes, full_edges = problemToSolve[0], problemToSolve[1]
        color_list = [str(item) for item in range(max(full_nodes)+1)]

        qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = True), shape = num_empty_fields)
        init_state = [random.choice(color_list) for _ in range(len(qarg))]

        def initial_state_mkcs(qarg):
            qarg[:] = init_state
            return qarg


        from qrisp import cp, cx, mcp
        def apply_phase_if_eq_qq(qcolor_0, qcolor_1, gamma):
            for i in range(qcolor_0.size):
                #If the i-th qubit is "1" for both variables they represent the same color
                #In this case the cp gate applies a phase of 2*gamma
                cp(2*gamma, qcolor_0[i], qcolor_1[i])


        def apply_phase_if_eq_cq(qcolor, color, gamma):
            #idea: apply phase gate to the qcolor[color] if it is == "1" as this then represents the same color as it its bordering nodecolor, which we control on
            #print(qcolor)
            #print(color)
            controlColor = QuantumColor(color_list, one_hot_enc = True)
            cx(qcolor[color], controlColor[color])
            #z(controlColor[color])
            # this dont work?... use ancilla qubit?
            with control(controlColor[color], "1"):
                p(2*gamma, qcolor[color])
            controlColor.uncompute()
            controlColor.delete()


        def create_coloring_operator(problem):
            nodes, edges = problem[0], problem[1]
            def coloring_operator(quantumcolor_array, gamma):
                for pair in edges:
                    if nodes[pair[0]] == -1 & nodes[pair[1]] == -1:
                        apply_phase_if_eq_qq(quantumcolor_array[eFI[pair[0]]], quantumcolor_array[eFI[pair[1]]], gamma)
                    elif nodes[pair[0]] == -1:
                        apply_phase_if_eq_cq(quantumcolor_array[eFI[pair[0]]], nodes[pair[1]], gamma)
                    elif nodes[pair[1]] == -1:
                        apply_phase_if_eq_cq(quantumcolor_array[eFI[pair[1]]], nodes[pair[0]], gamma)
            return coloring_operator


        def mkcs_obj(q_array, problem):

            
            quantumcolor_array =   [int(col) for line in q_array for col in line.split()]
            #print(quantumcolor_array)
            nodes, edges = problem[0], problem[1]
            color = 1
            for pair in edges:
                if nodes[pair[0]] == -1 & nodes[pair[1]] == -1:
                    if quantumcolor_array[eFI[pair[0]]] != quantumcolor_array[eFI[pair[1]]]:
                        color *= 4
                elif nodes[pair[0]] == -1:
                    if quantumcolor_array[eFI[pair[0]]] != nodes[pair[1]]:
                        color *= 4
                elif nodes[pair[1]] == -1:
                    if quantumcolor_array[eFI[pair[1]]] != nodes[pair[0]]:
                        color *= 4
            return -color


        def create_coloring_cl_cost_function(problem):
            def cl_cost_function(counts):
                energy = 0
                total_counts = 0
                
                for meas, meas_count in list(counts.items())[::-1]:

                    obj_for_meas = mkcs_obj(meas, problem)

                    energy += obj_for_meas * meas_count
                    total_counts += meas_count
                final_energy = (energy / total_counts)
                #print(final_energy)
                return energy / total_counts
            return cl_cost_function



        # we are here.....

        mkcs_onehot = QAOAProblem(create_coloring_operator(problemToSolve), apply_XY_mixer, create_coloring_cl_cost_function(problemToSolve))
        mkcs_onehot.set_init_function(lambda x : x.encode(init_state))
        
        #compile_qarg = QuantumArray(qtype = QuantumColor(color_list, one_hot_enc = True), shape = num_empty_fields)
        #qc, params= mkcs_onehot.compile_circuit(compile_qarg, depth=1)
        #print(qc)
        res_onehot = mkcs_onehot.run(qarg, depth= depth, max_iter = max_iter, mes_kwargs={"shots":shots})
        problemToSolve = problemToSolve
        #which one to pick??
        best_coloring, best_solution = min([(mkcs_obj(quantumcolor_array,problemToSolve),quantumcolor_array) for quantumcolor_array in res_onehot.keys()], key=itemgetter(0))
        #best_coloring_onehot, res_str_onehot = min([(mkcs_obj(quantumcolor_array,problemToSolve),quantumcolor_array) for quantumcolor_array in list(res_onehot.keys())[:5]], key=itemgetter(0))
        #best_coloring_onehot, best_solution_onehot = (mkcs_obj(res_str_onehot,problemToSolve),res_str_onehot)

        return_solution  =   [int(col) for line in best_solution for col in line.split()]
        print(return_solution)

        return return_solution, end_time_measurement(start), {}