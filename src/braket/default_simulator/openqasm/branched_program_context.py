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

from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
from sympy import Expr

from braket.default_simulator.gate_operations import BRAKET_GATES, GPhase, Unitary
from braket.default_simulator.noise_operations import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    Kraus,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
    TwoQubitDephasing,
    TwoQubitDepolarizing,
)
from braket.default_simulator.operation import GateOperation, Operation
from braket.ir.jaqcd.program_v1 import Results

from ._helpers.casting import LiteralType, get_identifier_name
from .circuit import Circuit
from .parser.openqasm_ast import (
    ClassicalType,
    FloatLiteral,
    Identifier,
    IndexedIdentifier,
    QuantumGateDefinition,
    SubroutineDefinition,
)
from .program_context import AbstractProgramContext, ProgramContext


class BranchedProgramContext(AbstractProgramContext):
    """
    A program context that manages multiple execution paths for branched simulation.
    
    This context maintains separate ProgramContext instances for each branch,
    allowing different variable values and circuit states per execution path.
    Instead of using circuits, it stores instruction sequences for each path.
    """

    def __init__(self, initial_context: Optional[AbstractProgramContext] = None):
        """
        Initialize the branched program context.

        Args:
            initial_context (Optional[AbstractProgramContext]): An initial context to copy.
                If None, creates a new ProgramContext.
        """
        # Don't call super().__init__() since we manage state per path
        
        # Create the initial branch context
        if initial_context is None:
            initial_context = ProgramContext()
        
        # Store contexts for each branch path
        self._branch_contexts: dict[int, AbstractProgramContext] = {0: initial_context}
        self._active_paths: list[int] = [0]
        
        # Store instruction sequences for each path
        self._instruction_sequences: dict[int, list[Operation]] = {0: []}
        
        # Initialize minimal state needed for the branched context
        self.num_qubits = initial_context.num_qubits
        self.inputs = initial_context.inputs.copy()

    @property
    def circuit(self) -> Circuit:
        """Circuit property not used in branched context - use instruction sequences instead."""
        raise NotImplementedError("BranchedProgramContext uses instruction sequences, not circuits. Use get_instruction_sequence_for_path() instead.")

    def get_instruction_sequence_for_path(self, path_id: int) -> list[Operation]:
        """Get the instruction sequence for a specific path."""
        if path_id not in self._instruction_sequences:
            raise ValueError(f"Path {path_id} does not exist")
        return self._instruction_sequences[path_id].copy()

    def get_active_paths(self) -> list[int]:
        """Get list of currently active path IDs."""
        return self._active_paths.copy()

    @property
    def qubit_mapping(self):
        """Get qubit mapping from first active path (for compatibility)."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].qubit_mapping

    def create_branch(self, source_path: int) -> int:
        """
        Create a new branch by copying an existing path.
        
        Args:
            source_path (int): The path ID to copy from.
            
        Returns:
            int: The new path ID.
        """
        if source_path not in self._branch_contexts:
            raise ValueError(f"Source path {source_path} does not exist")
        
        # Find next available path ID
        new_path_id = max(self._branch_contexts.keys()) + 1
        
        # Deep copy the source context
        source_context = self._branch_contexts[source_path]
        new_context = self._deep_copy_context(source_context)
        
        self._branch_contexts[new_path_id] = new_context
        self._active_paths.append(new_path_id)
        
        # Copy the instruction sequence
        self._instruction_sequences[new_path_id] = self._instruction_sequences[source_path].copy()
        
        return new_path_id

    def _deep_copy_context(self, context: AbstractProgramContext) -> AbstractProgramContext:
        """Create a deep copy of a program context."""
        # Create new context of the same type
        if isinstance(context, ProgramContext):
            new_context = ProgramContext()
        else:
            # For other context types, create a basic ProgramContext
            new_context = ProgramContext()
        
        # Copy all the tables and state
        new_context.symbol_table = deepcopy(context.symbol_table)
        new_context.variable_table = deepcopy(context.variable_table)
        new_context.gate_table = deepcopy(context.gate_table)
        new_context.subroutine_table = deepcopy(context.subroutine_table)
        new_context.qubit_mapping = deepcopy(context.qubit_mapping)
        new_context.inputs = deepcopy(context.inputs)
        new_context.num_qubits = context.num_qubits
        
        return new_context

    def remove_path(self, path_id: int) -> None:
        """Remove a path from active paths."""
        if path_id in self._active_paths:
            self._active_paths.remove(path_id)
        if path_id in self._branch_contexts:
            del self._branch_contexts[path_id]
        if path_id in self._instruction_sequences:
            del self._instruction_sequences[path_id]

    def is_builtin_gate(self, name: str) -> bool:
        """Check if a gate is a built-in Braket gate."""
        # Use first active path for gate checking (should be consistent across paths)
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].is_builtin_gate(name)

    def add_phase_instruction(self, target: tuple[int], phase_value: float) -> None:
        """Add phase instruction to all active paths' instruction sequences."""
        phase_instruction = GPhase(target, phase_value)
        for path_id in self._active_paths:
            self._instruction_sequences[path_id].append(phase_instruction)

    def add_gate_instruction(
        self, gate_name: str, target: tuple[int, ...], params, ctrl_modifiers: list[int], power: int
    ) -> None:
        """Add gate instruction to all active paths' instruction sequences."""
        instruction = BRAKET_GATES[gate_name](
            target, *params, ctrl_modifiers=ctrl_modifiers, power=power
        )
        for path_id in self._active_paths:
            self._instruction_sequences[path_id].append(instruction)

    def add_custom_unitary(self, unitary: np.ndarray, target: tuple[int, ...]) -> None:
        """Add custom unitary instruction to all active paths' instruction sequences."""
        instruction = Unitary(target, unitary)
        for path_id in self._active_paths:
            self._instruction_sequences[path_id].append(instruction)

    def add_noise_instruction(
        self, noise_instruction: str, target: list[int], probabilities: list[float]
    ) -> None:
        """Add noise instruction to all active paths' instruction sequences."""
        one_prob_noise_map = {
            "bit_flip": BitFlip,
            "phase_flip": PhaseFlip,
            "pauli_channel": PauliChannel,
            "depolarizing": Depolarizing,
            "two_qubit_depolarizing": TwoQubitDepolarizing,
            "two_qubit_dephasing": TwoQubitDephasing,
            "amplitude_damping": AmplitudeDamping,
            "generalized_amplitude_damping": GeneralizedAmplitudeDamping,
            "phase_damping": PhaseDamping,
        }
        instruction = one_prob_noise_map[noise_instruction](target, *probabilities)
        for path_id in self._active_paths:
            self._instruction_sequences[path_id].append(instruction)

    def add_kraus_instruction(self, matrices: list[np.ndarray], target: list[int]) -> None:
        """Add Kraus instruction to all active paths' instruction sequences."""
        instruction = Kraus(target, matrices)
        for path_id in self._active_paths:
            self._instruction_sequences[path_id].append(instruction)

    def add_result(self, result: Results) -> None:
        """Add result to all active paths' contexts."""
        for path_id in self._active_paths:
            self._branch_contexts[path_id].add_result(result)

    def add_measure(self, target: tuple[int], classical_targets: Optional[Iterable[int]] = None) -> None:
        """Add measurement with branching based on quantum state probabilities."""
        from braket.default_simulator.state_vector_simulation import StateVectorSimulation
        
        # Get current active paths
        original_active_paths = self._active_paths.copy()
        paths_to_remove = []
        
        # Process each active path
        for path_id in original_active_paths:
            try:
                # Get the instruction sequence for this path
                instructions = self._instruction_sequences[path_id]
                
                # Create a simulation to evolve the state
                sim = StateVectorSimulation(self.num_qubits, shots=1, batch_size=1)
                
                # Evolve the state with all instructions up to this point
                if instructions:
                    sim.evolve(instructions)
                
                # Calculate measurement probabilities for the target qubit
                # For simplicity, assume single qubit measurement for now
                if len(target) == 1:
                    qubit_idx = target[0]
                    probs = self._get_measurement_probabilities(sim.state_vector, qubit_idx)
                    
                    # Determine if we need to branch based on probabilities
                    prob_0, prob_1 = probs[0], probs[1]
                    
                    # If both outcomes are possible (neither probability is 0), create branches
                    if prob_0 > 1e-10 and prob_1 > 1e-10:
                        # Create a new path for outcome 1
                        new_path_id = self.create_branch(path_id)
                        
                        # Update classical variables for each outcome
                        self._update_measurement_outcome(path_id, classical_targets, 0)
                        self._update_measurement_outcome(new_path_id, classical_targets, 1)
                            
                        print(f"Branched path {path_id} -> outcomes 0 (path {path_id}) and 1 (path {new_path_id})")
                        print(f"Probabilities: P(0)={prob_0:.3f}, P(1)={prob_1:.3f}")
                        
                    else:
                        # Deterministic outcome
                        outcome = 0 if prob_1 < 1e-10 else 1
                        self._update_measurement_outcome(path_id, classical_targets, outcome)
                        print(f"Deterministic measurement on path {path_id}: outcome {outcome}")
                
                else:
                    # Multi-qubit measurement - for now, just use the first qubit
                    # TODO: Implement proper multi-qubit measurement
                    print("Multi-qubit measurement not fully implemented, using first qubit")
                    self._update_measurement_outcome(path_id, classical_targets, 0)
                        
            except Exception as e:  # noqa: PERF203
                print(f"Error processing measurement for path {path_id}: {e}")
                import traceback
                traceback.print_exc()
                paths_to_remove.append(path_id)
        
        # Remove any failed paths
        for path_id in paths_to_remove:
            self.remove_path(path_id)

    def _get_measurement_probabilities(self, state_vector: np.ndarray, qubit_idx: int) -> np.ndarray:
        """Calculate measurement probabilities for a specific qubit"""
        # Reshape state to tensor form
        state_tensor = np.reshape(state_vector, [2] * self.num_qubits)
        
        # Extract slices for |0⟩ and |1⟩ states of the target qubit
        slice_0 = np.take(state_tensor, 0, axis=qubit_idx)
        slice_1 = np.take(state_tensor, 1, axis=qubit_idx)
        
        # Calculate probabilities
        prob_0 = np.sum(np.abs(slice_0) ** 2)
        prob_1 = np.sum(np.abs(slice_1) ** 2)
        
        return np.array([prob_0, prob_1])
    
    def _update_measurement_outcome(self, path_id: int, classical_targets: Optional[Iterable[int]], outcome: int):
        """Update classical variables with measurement outcome for a specific path"""
        if classical_targets is not None:
            # Add the measurement to the specific path's context
            self._branch_contexts[path_id].add_measure((outcome,), classical_targets)
        
        # Note: The actual classical variable update would need to be handled
        # by the interpreter when it processes the measurement statement

    # Override methods that need to work with all active paths' contexts
    def declare_variable(
        self,
        name: str,
        symbol_type: Union[ClassicalType, type[LiteralType], type[Identifier]],
        value: Optional[Any] = None,
        const: bool = False,
    ) -> None:
        """Declare variable in all active paths' contexts."""
        for path_id in self._active_paths:
            self._branch_contexts[path_id].declare_variable(name, symbol_type, value, const)

    def declare_qubit_alias(self, name: str, value: Identifier) -> None:
        """Declare qubit alias in all active paths' contexts."""
        for path_id in self._active_paths:
            self._branch_contexts[path_id].declare_qubit_alias(name, value)

    def push_scope(self) -> None:
        """Push scope in all active paths' contexts."""
        for path_id in self._active_paths:
            self._branch_contexts[path_id].push_scope()

    def pop_scope(self) -> None:
        """Pop scope in all active paths' contexts."""
        for path_id in self._active_paths:
            self._branch_contexts[path_id].pop_scope()

    @property
    def in_global_scope(self) -> bool:
        """Check if first active path is in global scope."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].in_global_scope

    def get_type(self, name: str) -> Union[ClassicalType, type[LiteralType]]:
        """Get symbol type from first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].get_type(name)

    def get_const(self, name: str) -> bool:
        """Get const status from first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].get_const(name)

    def get_value(self, name: str) -> LiteralType:
        """Get value from first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].get_value(name)

    def get_value_by_identifier(
        self, identifier: Union[Identifier, IndexedIdentifier]
    ) -> LiteralType:
        """Get value by identifier from first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].get_value_by_identifier(identifier)

    def is_initialized(self, name: str) -> bool:
        """Check if variable is initialized in first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].is_initialized(name)

    def update_value(self, variable: Union[Identifier, IndexedIdentifier], value: Any) -> None:
        """Update value in all active paths' contexts."""
        for path_id in self._active_paths:
            self._branch_contexts[path_id].update_value(variable, value)

    def add_qubits(self, name: str, num_qubits: Optional[int] = 1) -> None:
        """Add qubits to all path contexts."""
        if num_qubits is None:
            num_qubits = 1
        for context in self._branch_contexts.values():
            context.add_qubits(name, num_qubits)
        self.num_qubits += num_qubits

    def get_qubits(self, qubits: Union[Identifier, IndexedIdentifier]) -> tuple[int]:
        """Get qubit indices from first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].get_qubits(qubits)

    def add_gate(self, name: str, definition: QuantumGateDefinition) -> None:
        """Add gate definition to all active paths' contexts."""
        for path_id in self._active_paths:
            self._branch_contexts[path_id].add_gate(name, definition)

    def get_gate_definition(self, name: str) -> QuantumGateDefinition:
        """Get gate definition from first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].get_gate_definition(name)

    def is_user_defined_gate(self, name: str) -> bool:
        """Check if gate is user-defined in first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].is_user_defined_gate(name)

    def add_subroutine(self, name: str, definition: SubroutineDefinition) -> None:
        """Add subroutine definition to all active paths' contexts."""
        for path_id in self._active_paths:
            self._branch_contexts[path_id].add_subroutine(name, definition)

    def get_subroutine_definition(self, name: str) -> SubroutineDefinition:
        """Get subroutine definition from first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].get_subroutine_definition(name)

    def handle_parameter_value(self, value: Union[float, Expr]) -> Any:
        """Handle parameter value using first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].handle_parameter_value(value)

    def load_inputs(self, inputs: dict[str, Any]) -> None:
        """Load inputs into all path contexts."""
        self.inputs.update(inputs)
        for context in self._branch_contexts.values():
            context.load_inputs(inputs)

    def parse_pragma(self, pragma_body: str):
        """Parse pragma using first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].parse_pragma(pragma_body)

    def enter_scope(self):
        """Enter scope in first active path's context."""
        if not self._active_paths:
            raise RuntimeError("No active paths available")
        return self._branch_contexts[self._active_paths[0]].enter_scope()

    def __repr__(self):
        """String representation showing all branch contexts."""
        lines = [f"BranchedProgramContext with {len(self._branch_contexts)} paths:"]
        lines.append(f"Active paths: {self._active_paths}")
        
        for path_id, context in self._branch_contexts.items():
            lines.append(f"\n--- Path {path_id} ---")
            lines.append(f"Instructions: {len(self._instruction_sequences[path_id])}")
            lines.append(str(context))
        
        return "\n".join(lines)
