#!/usr/bin/env python3
"""
Simple test to verify BranchedProgramContext and BranchedInterpreter work.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from braket.default_simulator.openqasm.branched_program_context import BranchedProgramContext
from braket.default_simulator.openqasm.branched_interpreter import BranchedInterpreter


def test_basic_program_context():
    """Test basic BranchedProgramContext functionality"""
    print("Testing BranchedProgramContext...")

    # Create a branched context
    context = BranchedProgramContext()

    # Test basic properties
    print(f"Initial active paths: {context.get_active_paths()}")
    print(f"Initial num_qubits: {context.num_qubits}")

    # Test adding qubits
    context.add_qubits("q", 2)
    print(f"After adding 2 qubits: {context.num_qubits}")

    # Test creating a branch
    new_path = context.create_branch(0)
    print(f"Created new branch: {new_path}")
    print(f"Active paths after branching: {context.get_active_paths()}")

    # Test instruction sequences
    for path_id in context.get_active_paths():
        instructions = context.get_instruction_sequence_for_path(path_id)
        print(f"Path {path_id} has {len(instructions)} instructions")

    print("BranchedProgramContext basic test passed!\n")


def test_simple_qasm_program():
    """Test with a simple OpenQASM program"""
    print("Testing simple OpenQASM program...")

    # Simple program: declare qubits and apply a gate
    qasm_program = """
    OPENQASM 3.0;
    qubit[2] q;
    h q[0];
    """

    try:
        interpreter = BranchedInterpreter()
        print("Created BranchedInterpreter")

        # Try to run the program
        result_context = interpreter.run(qasm_program)
        print("Program executed successfully!")

        print(f"Final active paths: {result_context.get_active_paths()}")
        print(f"Final num_qubits: {result_context.num_qubits}")

        # Check instruction sequences
        for path_id in result_context.get_active_paths():
            instructions = result_context.get_instruction_sequence_for_path(path_id)
            print(f"Path {path_id} has {len(instructions)} instructions")
            for i, instr in enumerate(instructions):
                print(f"  Instruction {i}: {type(instr).__name__}")

    except Exception as e:
        print(f"Error running simple program: {e}")
        import traceback

        traceback.print_exc()

    print("Simple OpenQASM test completed!\n")


def test_branching_program():
    """Test with a program that should cause branching"""
    print("Testing branching program...")

    # Program with classical variable and conditional
    qasm_program = """
    OPENQASM 3.0;
    qubit q;
    bit c;
    h q;
    c = measure q;
    if (c) {
        x q;
    }
    """

    try:
        interpreter = BranchedInterpreter()
        print("Created BranchedInterpreter for branching test")

        # Try to run the program
        result_context = interpreter.run(qasm_program)
        print("Branching program executed!")

        print(f"Final active paths: {result_context.get_active_paths()}")

        # Check instruction sequences for each path
        for path_id in result_context.get_active_paths():
            instructions = result_context.get_instruction_sequence_for_path(path_id)
            print(f"Path {path_id} has {len(instructions)} instructions:")
            for i, instr in enumerate(instructions):
                print(f"  Instruction {i}: {type(instr).__name__}")

    except Exception as e:
        print(f"Error running branching program: {e}")
        import traceback

        traceback.print_exc()

    print("Branching program test completed!\n")


if __name__ == "__main__":
    print("Starting BranchedProgramContext and BranchedInterpreter tests...\n")

    test_basic_program_context()
    test_simple_qasm_program()
    test_branching_program()

    print("All tests completed!")
