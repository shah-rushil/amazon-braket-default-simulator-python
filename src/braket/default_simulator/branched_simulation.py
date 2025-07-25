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

import numpy as np
from typing import Dict, List, Any, Optional, Union
from copy import deepcopy

from braket.default_simulator.operation import GateOperation, Observable
from braket.default_simulator.simulation import Simulation
from braket.default_simulator.state_vector_simulation import StateVectorSimulation
from braket.default_simulator.gate_operations import Measure
from braket.default_simulator.gate_operations import (
    Identity, Hadamard, PauliX, PauliY, PauliZ, CX, RotX, RotY, RotZ, S, T
)

# Additional structures for advanced features
class GateDefinition:
    """Store custom gate definitions."""
    def __init__(self, name: str, arguments: List[str], qubit_targets: List[str], body: Any):
        self.name = name
        self.arguments = arguments
        self.qubit_targets = qubit_targets
        self.body = body

class FunctionDefinition:
    """Store custom function definitions."""
    def __init__(self, name: str, arguments: Any, body: List[Any], return_type: Any):
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
    
    def __repr__(self):
        return f"FramedVariable(name='{self.name}', type={self.type}, val={self.val}, is_const={self.is_const}, frame={self.frame_number})"

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
        if shots <= 0:
            raise ValueError("BranchedSimulation requires shots > 0")
            
        super().__init__(qubit_count=qubit_count, shots=shots)
        
        # Core branching state
        self._batch_size = batch_size
        self._instruction_sequences: List[List[GateOperation]] = [[]]
        self._active_paths: List[int] = [0]
        self._shots_per_path: List[int] = [shots]
        self._measurements: List[Dict[int, List[int]]] = [{}]  # path_idx -> {qubit_idx: [outcomes]}
        self._variables: List[Dict[str, FramedVariable]] = [{}]  # Classical variables per path
        self._curr_frame: int = 0 # Variable Frame

        # Return values for function calls
        self._return_values: Dict[int, Any] = {}

        # Simulation indices for continue in for loop
        self._continue_paths: List[int] = []
        
        # Qubit management
        self._qubit_mapping: Dict[str, Union[int, List[int]]] = {}
        self._measured_qubits: List[int] = []
        
        # No state caching - always calculate fresh to avoid exponential memory growth

    def evolve(self, operations: List[GateOperation]) -> None:
        """
        Add operations to all active paths.
        This doesn't execute them immediately - they're stored for lazy evaluation.
        """
        for path_idx in self._active_paths:
            self._instruction_sequences[path_idx].extend(operations)

    def measure_qubit_on_path(self, path_idx: int, qubit_idx: int, qubit_name: Optional[str] = None) -> None:
        """
        Perform measurement on a qubit for a specific path.
        Returns the new path indices that result from this measurement.
        Optimized to avoid unnecessary branching when outcome is deterministic.
        """
        if path_idx not in self._active_paths:
            return None
        
        # Calculate current state for this path
        current_state = self._get_path_state(path_idx)
        
        # Get measurement probabilities
        probs = self._get_measurement_probabilities(current_state, qubit_idx)
        
        path_shots = self._shots_per_path[path_idx]
        shots_for_outcome_0 = round(path_shots*probs[0])
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
            
    def _get_path_state(self, path_idx: int) -> np.ndarray:
        """
        Get the current state for a specific path by calculating it fresh from the instruction sequence.
        No caching is used to avoid exponential memory growth.
        """
        # Create a fresh StateVectorSimulation and apply all operations
        sim = StateVectorSimulation(self._qubit_count, self._shots_per_path[path_idx], self._batch_size)
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
        prob_0 = np.sum(np.abs(slice_0)**2)
        prob_1 = np.sum(np.abs(slice_1)**2)
        
        return np.array([prob_0, prob_1])

    def retrieve_samples(self) -> List[int]:
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
                    len(probabilities), 
                    size=path_shots, 
                    p=probabilities
                )
                
                all_samples.extend(path_samples.tolist())
        
        return all_samples

    def get_measurement_counts(self) -> Dict[str, int]:
        """
        Get measurement counts aggregated across all paths.
        Returns a dictionary mapping bit strings to counts.
        """
        counts = {}
        
        for path_idx in self._active_paths:
            path_shots = self._shots_per_path[path_idx]
            if path_shots > 0:
                # Build bit string from measurements
                bit_string = ""
                for qubit_idx in sorted(self._measured_qubits):
                    if qubit_idx in self._measurements[path_idx]:
                        # Use the last measurement outcome for this qubit
                        outcome = self._measurements[path_idx][qubit_idx][-1]
                        bit_string += str(outcome)
                    else:
                        bit_string += "0"  # Default to 0 if not measured
                
                if bit_string in counts:
                    counts[bit_string] += path_shots
                else:
                    counts[bit_string] = path_shots
        
        return counts

    def set_variable(self, path_idx: int, var_name: str, value: FramedVariable) -> None:
        """Set a classical variable for a specific path."""
        if path_idx < len(self._variables):
            self._variables[path_idx][var_name] = value

    def get_variable(self, path_idx: int, var_name: str, default: Any = None) -> Any:
        """Get a classical variable for a specific path."""
        if path_idx < len(self._variables):
            return self._variables[path_idx].get(var_name, default)
        return default

    def add_qubit_mapping(self, name: str, indices: Union[int, List[int]]) -> None:
        """Add a mapping from qubit name to indices."""
        self._qubit_mapping[name] = indices
        self._qubit_count += 1

    def get_qubit_indices(self, name: str) -> Union[int, List[int]]:
        """Get qubit indices for a given name."""
        return self._qubit_mapping.get(name)

    @property
    def num_paths(self) -> int:
        """Get total number of paths."""
        return len(self._instruction_sequences)

    @property
    def measured_qubits(self) -> List[int]:
        """Get list of measured qubit indices."""
        return self._measured_qubits.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """
        Get probabilities aggregated across all paths.
        This is a weighted average based on shots per path.
        """
        if not self._active_paths:
            return np.array([])
        
        total_shots = sum(self._shots_per_path[i] for i in self._active_paths)
        if total_shots == 0:
            return np.array([])
        
        # Aggregate probabilities weighted by shots
        aggregated_probs = np.zeros(2**self._qubit_count)
        
        for path_idx in self._active_paths:
            path_shots = self._shots_per_path[path_idx]
            if path_shots > 0:
                state = self._get_path_state(path_idx)
                path_probs = np.abs(state) ** 2
                weight = path_shots / total_shots
                aggregated_probs += weight * path_probs
        
        return aggregated_probs

    @property
    def state_vector(self) -> np.ndarray:
        """
        Get aggregated state vector across all paths.
        This is a weighted superposition based on shots per path.
        """
        if not self._active_paths:
            return np.zeros(2**self._qubit_count, dtype=complex)
        
        total_shots = sum(self._shots_per_path[i] for i in self._active_paths)
        if total_shots == 0:
            return np.zeros(2**self._qubit_count, dtype=complex)
        
        # Aggregate state vectors weighted by sqrt(shots)
        aggregated_state = np.zeros(2**self._qubit_count, dtype=complex)
        
        for path_idx in self._active_paths:
            path_shots = self._shots_per_path[path_idx]
            if path_shots > 0:
                state = self._get_path_state(path_idx)
                weight = np.sqrt(path_shots / total_shots)
                aggregated_state += weight * state
        
        # Normalize
        norm = np.linalg.norm(aggregated_state)
        if norm > 0:
            aggregated_state /= norm
        
        return aggregated_state
