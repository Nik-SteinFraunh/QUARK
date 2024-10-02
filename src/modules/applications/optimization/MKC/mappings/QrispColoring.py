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

import networkx
import numpy as np
import pulser

from modules.applications.Mapping import *
from utils import start_time_measurement, end_time_measurement


class QrispColoring(Mapping):
    """
    Qrisp graph coloring formulation for MKC.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.submodule_options = ["BacktrackingColoring", "QAOAColoring"]

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
                "version": "0.4.12"
           }
        ]

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {}

        """
        return {}

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python
            pass
        """
        pass

    def map(self, problem: list, config: Config) -> (dict, float):
        """
        Maps the networkx graph to a neutral atom MIS problem.

        :param problem: networkx graph
        :type problem: networkx.Graph
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with neutral MIS, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = start_time_measurement()


        backtracking_problem = {
            'sudoku': problem,
        }
        return backtracking_problem, end_time_measurement(start)

    def get_default_submodule(self, option: str) -> Core:

        if option == "BacktrackingColoring":
            from modules.solvers.QRISPBacktrackingSolver import QRISPBacktrackingSolver  # pylint: disable=C0415
            return QRISPBacktrackingSolver()
        if option == "QAOAColoring":
            from modules.solvers.QRISPQAOASolverMKC import QRISPQAOASolverMKC  # pylint: disable=C0415
            return QRISPQAOASolverMKC()
        else:
            raise NotImplementedError(f"Solver Option {option} not implemented")
