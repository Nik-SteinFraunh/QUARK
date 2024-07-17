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
from abc import ABC, abstractmethod
import time

try:
    import cupy as np
    GPU = True
    logging.info("Using CuPy, data processing on GPU")
except ModuleNotFoundError:
    import numpy as np
    GPU = False
    logging.info("CuPy not available, using vanilla numpy, data processing on CPU")

from modules.Core import Core
from utils import start_time_measurement, end_time_measurement


class Training(Core, ABC):
    """
    The Training module is the base class fot both finding (QCBM) and executing trained models (Inference)
    """

    def __init__(self, name):
        """
        Constructor method
        """
        self.name = name
        super().__init__()
        self.n_states_range = None

    @staticmethod
    def get_requirements() -> list[dict]:
        """
        Returns requirements of this module

        :return: list of dict with requirements of this module
        :rtype: list[dict]
        """
        return [
            {
                "name": "numpy",
                "version": "1.23.5"
            }
        ]

    def postprocess(self, input_data: dict, config: dict, **kwargs):
        """
        Here, the actual training of the machine learning model is done

        :param input_data: Collected information of the benchmarking process
        :type input_data: dict
        :param config: Training settings
        :type config: dict
        :param kwargs: Optional additional arguments
        :type kwargs: dict
        :return: 
        :rtype: 
        """
        start = start_time_measurement()
        logging.info("Start training")
        training_results = self.start_training(
            self.preprocessed_input,
            config,
            **kwargs
        )
        for dict_key in ["backend", "circuit", "execute_circuit"]:
            training_results.pop(dict_key)
        postprocessing_time = end_time_measurement(start)
        logging.info(f"Training finished in {postprocessing_time / 1000} s.")
        return training_results, postprocessing_time

    @abstractmethod
    def start_training(self, input_data: dict, config: any, **kwargs: dict) -> dict:
        """
        This function starts the training of QML model or deploys a pretrained model.

        :param input_data: A representation of the quantum machine learning model that will be trained
        :type input_data: dict
        :param config: Config specifying the parameters of the training (dict-like Config type defined in children)
        :type config: any
        :param kwargs: optional additional settings
        :type kwargs: dict
        :return: Solution, the time it took to compute it and some optional additional information
        :rtype: dict
        """
        pass

    def sample_from_pmf(self, pmf: np.ndarray, n_shots: int) -> np.ndarray:
        """
        Function to sample from the probability mass function generated by the quantum circuit

        :param pmf: Probability mass function generated by the quantum circuit
        :type pmf: np.ndarray
        :param n_shots: Number of shots
        :type n_shots: int
        :return: number of counts in the 2**n_qubits bins
        :rtype: np.ndarray
        """
        samples = np.random.choice(self.n_states_range, size=n_shots, p=pmf)
        counts = np.bincount(samples, minlength=len(self.n_states_range))
        return counts

    def kl_divergence(self, pmf_model: np.ndarray, pmf_target: np.ndarray) -> float:
        """
        Kullback-Leibler divergence, that is used as a loss function

        :param pmf_model: Probability mass function generated by the quantum circuit
        :type pmf_model: np.ndarray
        :param pmf_target: Probability mass function of the target distribution
        :type pmf_target: np.ndarray
        :return: Kullback-Leibler divergence
        :rtype: float
        """
        pmf_model[pmf_model == 0] = 1e-8
        return np.sum(pmf_target * np.log(pmf_target / pmf_model), axis=1)

    def nll(self, pmf_model: np.ndarray, pmf_target: np.ndarray) -> float:
        """
        Negative log likelihood, that is used as a loss function

        :param pmf_model: Probability mass function generated by the quantum circuit
        :type pmf_model: np.ndarray
        :param pmf_target: Probability mass function of the target distribution
        :type pmf_target: np.ndarray
        :return: Negative log likelihood
        :rtype: float
        """
        pmf_model[pmf_model == 0] = 1e-8
        return -np.sum(pmf_target * np.log(pmf_model), axis=1)

    def mmd(self, pmf_model: np.ndarray, pmf_target: np.ndarray) -> float:
        """
        Maximum mean discrepancy, that is used as a loss function

        :param pmf_model: Probability mass function generated by the quantum circuit
        :type pmf_model: np.ndarray
        :param pmf_target: Probability mass function of the target distribution
        :type pmf_target: np.ndarray
        :return: Maximum mean discrepancy
        :rtype: float
        """
        pmf_model[pmf_model == 0] = 1e-8
        sigma = 1/pmf_model.shape[1]
        kernel_distance = np.exp((-np.square(pmf_model - pmf_target) / (sigma ** 2)))
        mmd = 2 - 2 * np.mean(kernel_distance, axis=1)
        return mmd

    class Timing:
        """
        This module is an abstraction of time measurement for both CPU and GPU processes
        """

        def __init__(self):
            """
            Constructor method
            """

            if GPU:
                self.start_cpu: time.perf_counter
            else:
                self.start_gpu: np.cuda.Event
                self.end_gpu: time.perf_counter

            self.start_recording = self.start_recording_gpu if GPU else self.start_recording_cpu
            self.stop_recording = self.stop_recording_gpu if GPU else self.stop_recording_cpu

        def start_recording_cpu(self):
            """
            Function to start time measurement on the CPU
            """
            self.start_cpu = start_time_measurement()

        def stop_recording_cpu(self):
            """
            Function to stop time measurement on the CPU
            """
            return end_time_measurement(self.start_cpu)

        def start_recording_gpu(self):
            """
            Function to start time measurement on the GPU
            """
            self.start_gpu = np.cuda.Event()
            self.end_gpu = np.cuda.Event()
            self.start_gpu.record()

        def stop_recording_gpu(self):
            """
            Function to stop time measurement on the GPU
            """
            self.end_gpu.record()
            self.end_gpu.synchronize()
            return np.cuda.get_elapsed_time(self.start_gpu, self.end_gpu)
