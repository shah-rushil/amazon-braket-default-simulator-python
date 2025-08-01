# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np

from braket.default_simulator.gate_operations import Measure
from braket.default_simulator.operation import GateOperation
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.state_vector_simulation import StateVectorSimulation


# Additional structures for advanced features
class GateDefinition:
    """Store custom gate definitions."""

    def __init__(self, name: str, arguments: list[str], qubit_targets: list[str], body: Any):
        self.name = name
        self.arguments = arguments
        self.qubit_targets = qubit_targets
        self.body = body


class FunctionDefinition:
    """Store custom function definitions."""

    def __init__(self, name: str, arguments: Any, body: list[Any], return_type: Any):
        self.name = name
        self.arguments = arguments
        self.body = body
        self.return_type = return_type


class FramedVariable:
    """Variable with frame tracking for proper scoping."""

    def __init__(self, name: str, var_type: Any, value: Any, is_const: bool, frame_number: int):
        self.name = name
        self.type = var_type
        self.val = value
        self.is_const = is_const
        self.frame_number = frame_number


class BranchedSimulation(Simulation):
    """
    A simulation that supports multiple execution paths resulting from mid-circuit measurements.

    This class manages multiple StateVectorSimulation instances, one for each execution path.
    When a measurement occurs, paths may branch based on the measurement probabilities.
    """

    def __init__(self, qubit_count: int, shots: int, batch_size: int):
        """
        Initialize branched simulation.

        Args:
            qubit_count (int): The number of qubits being simulated.
            shots (int): The number of samples to take from the simulation. Must be > 0.
            batch_size (int): The size of the partitions to contract.
        """

        super().__init__(qubit_count=qubit_count, shots=shots)

        # Core branching state
        self._batch_size = batch_size
        self._instruction_sequences: list[list[GateOperation]] = [[]]
        self._active_paths: list[int] = [0]
        self._shots_per_path: list[int] = [shots]
        self._measurements: list[dict[int, list[int]]] = [{}]  # path_idx -> {qubit_idx: [outcomes]}
        self._variables: list[dict[str, FramedVariable]] = [{}]  # Classical variables per path
        self._curr_frame: int = 0  # Variable Frame

        # Return values for function calls
        self._return_values: dict[int, Any] = {}

        # Simulation indices for continue in for loop
        self._continue_paths: list[int] = []

        # Qubit management
        self._qubit_mapping: dict[str, Union[int, list[int]]] = {}
        self._measured_qubits: list[int] = []

    def measure_qubit_on_path(
        self, path_idx: int, qubit_idx: int, qubit_name: Optional[str] = None
    ) -> int:
        """
        Perform measurement on a qubit for a specific path.
        Returns the new path indices that result from this measurement.
        Optimized to avoid unnecessary branching when outcome is deterministic.
        """

        # Calculate current state for this path
        current_state = self._get_path_state(path_idx)

        # Get measurement probabilities
        probs = self._get_measurement_probabilities(current_state, qubit_idx)

        path_shots = self._shots_per_path[path_idx]
        shots_for_outcome_0 = round(path_shots * probs[0])
        shots_for_outcome_1 = path_shots - shots_for_outcome_0

        if shots_for_outcome_1 == 0 or shots_for_outcome_0 == 0:
            # Deterministic outcome 0 - no need to branch
            outcome = 0 if shots_for_outcome_1 == 0 else 1

            # Update the existing path in place
            measure_op = Measure([qubit_idx], result=outcome)
            self._instruction_sequences[path_idx].append(measure_op)

            if qubit_idx not in self._measurements[path_idx]:
                self._measurements[path_idx][qubit_idx] = []
            self._measurements[path_idx][qubit_idx].append(outcome)

            # Track measured qubits
            if qubit_idx not in self._measured_qubits:
                self._measured_qubits.append(qubit_idx)

            return -1

        else:
            # Path for outcome 0
            path_0_instructions = self._instruction_sequences[path_idx]
            path_1_instructions = path_0_instructions.copy()

            measure_op_0 = Measure([qubit_idx], result=0)
            path_0_instructions.append(measure_op_0)

            self._shots_per_path[path_idx] = shots_for_outcome_0
            new_measurements_0 = self._measurements[path_idx]
            new_measurements_1 = deepcopy(self._measurements[path_idx])

            if qubit_idx not in new_measurements_0:
                new_measurements_0[qubit_idx] = []
            new_measurements_0[qubit_idx].append(0)

            # Path for outcome 1
            path_1_idx = len(self._instruction_sequences)
            measure_op_1 = Measure([qubit_idx], result=1)
            path_1_instructions.append(measure_op_1)
            self._instruction_sequences.append(path_1_instructions)
            self._shots_per_path.append(shots_for_outcome_1)

            if qubit_idx not in new_measurements_1:
                new_measurements_1[qubit_idx] = []
            new_measurements_1[qubit_idx].append(1)
            self._measurements.append(new_measurements_1)
            self._variables.append(deepcopy(self._variables[path_idx]))

            # Add new paths to active paths
            self._active_paths.append(path_1_idx)

            return path_1_idx

    def _get_path_state(self, path_idx: int) -> np.ndarray:
        """
        Get the current state for a specific path by calculating it fresh from the instruction sequence.
        No caching is used to avoid exponential memory growth.
        """
        # Create a fresh StateVectorSimulation and apply all operations
        sim = StateVectorSimulation(
            self._qubit_count, self._shots_per_path[path_idx], self._batch_size
        )
        sim.evolve(self._instruction_sequences[path_idx])

        return sim.state_vector

    def _get_measurement_probabilities(self, state: np.ndarray, qubit_idx: int) -> np.ndarray:
        """
        Calculate measurement probabilities for a specific qubit using little-endian convention.

        In little-endian: for state |10⟩, qubit 0 is |1⟩ and qubit 1 is |0⟩.
        The tensor axes are ordered such that qubit 0 is the rightmost (last) axis.
        """
        # Reshape state to tensor form with little-endian qubit ordering
        # qubit 0 is the last axis, qubit 1 is second-to-last, etc.
        state_tensor = np.reshape(state, [2] * self._qubit_count)

        # Extract slices for |0⟩ and |1⟩ states of the target qubit
        slice_0 = np.take(state_tensor, 0, axis=qubit_idx)
        slice_1 = np.take(state_tensor, 1, axis=qubit_idx)

        # Calculate probabilities by summing over all remaining dimensions
        # After np.take(), we have one fewer dimension, so sum over all remaining axes
        prob_0 = np.sum(np.abs(slice_0) ** 2)
        prob_1 = np.sum(np.abs(slice_1) ** 2)

        return np.array([prob_0, prob_1])

    def retrieve_samples(self) -> list[int]:
        """
        Retrieve samples by aggregating across all paths.
        Calculate final state for each path and sample from it directly.
        """
        all_samples = []

        for path_idx in self._active_paths:
            path_shots = self._shots_per_path[path_idx]
            if path_shots > 0:
                # Calculate the final state once for this path
                final_state = self._get_path_state(path_idx)

                # Calculate probabilities for all possible outcomes
                probabilities = np.abs(final_state) ** 2

                # Sample from the probability distribution
                rng_generator = np.random.default_rng()
                path_samples = rng_generator.choice(
                    len(probabilities), size=path_shots, p=probabilities
                )

                all_samples.extend(path_samples.tolist())

        return all_samples

    def set_variable(self, path_idx: int, var_name: str, value: FramedVariable) -> None:
        """Set a classical variable for a specific path."""
        self._variables[path_idx][var_name] = value

    def get_variable(self, path_idx: int, var_name: str, default: Any = None) -> Any:
        """Get a classical variable for a specific path."""
        return self._variables[path_idx].get(var_name, default)

    def add_qubit_mapping(self, name: str, indices: Union[int, list[int]]) -> None:
        """Add a mapping from qubit name to indices."""
        self._qubit_mapping[name] = indices
        # Update qubit count based on the maximum index used
        if isinstance(indices, list):
            self._qubit_count += len(indices)
        else:
            self._qubit_count += 1

    def get_qubit_indices(self, name: str) -> Union[int, list[int]]:
        """Get qubit indices for a given name."""
        return self._qubit_mapping[name]

    def get_current_state_vector(self, path_idx: int) -> np.ndarray:
        """Get the current state vector for a specific path."""
        return self._get_path_state(path_idx)

    def load_from_branched_context(self, context):
        """Load instruction sequences from a BranchedProgramContext"""
        active_paths = context.get_active_paths()
        
        # Clear existing state
        self._instruction_sequences = []
        self._active_paths = []
        self._shots_per_path = []
        self._measurements = []
        self._variables = []
        
        # Update qubit count
        self._qubit_count = context.num_qubits
        
        # Calculate shots per path
        shots_per_path = self._shots // len(active_paths) if active_paths else self._shots
        remaining_shots = self._shots % len(active_paths) if active_paths else 0
        
        # Load each path
        for i, path_id in enumerate(active_paths):
            try:
                # Get instruction sequence for this path
                instructions = context.get_instruction_sequence_for_path(path_id)
                
                # Calculate shots for this path
                path_shots = shots_per_path + (1 if i < remaining_shots else 0)
                
                # Add to simulation
                self._instruction_sequences.append(instructions)
                self._active_paths.append(len(self._instruction_sequences) - 1)
                self._shots_per_path.append(path_shots)
                self._measurements.append({})  # Empty measurements dict
                self._variables.append({})  # Empty variables dict
                
            except Exception as e:
                print(f"Error loading path {path_id}: {e}")
                # Add empty path for failed paths
                self._instruction_sequences.append([])
                self._active_paths.append(len(self._instruction_sequences) - 1)
                self._shots_per_path.append(0)
                self._measurements.append({})
                self._variables.append({})

    def get_measurements_as_arrays(self):
        """Get measurements in the format expected by the simulator (list of arrays)"""
        samples = self.retrieve_samples()
        measurements = []
        
        for sample in samples:
            # Convert to binary representation
            binary_str = format(sample, f'0{self._qubit_count}b')
            # Convert to array of integers (little-endian: rightmost bit is qubit 0)
            measurement = [int(bit) for bit in reversed(binary_str)]
            measurements.append(measurement)
            
        return measurements
