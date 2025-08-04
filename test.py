from collections import Counter

import numpy as np

from braket.default_simulator.branched_simulator import BranchedSimulator
from braket.ir.openqasm import Program as OpenQASMProgram

qasm_source = """
        OPENQASM 3.0;
        qubit[4] q;
        bit[4] b;
        int[32] sum;

        // Initialize all qubits to |+⟩ state
        for uint i in [0:3] {
            h q[i];
        }

        // Measure all qubits
        for uint i in [0:3] {
            b[i] = measure q[i];
        }

        // Count the number of 1s measured
        for uint i in [0:3] {
            if (b[i] == 1) {
                sum = sum + 1;
            }
        }

        // Apply operations based on the sum
        if (sum == 1){
            x q[0];  // Apply X to qubit 0
        }
        if (sum == 2){
            h q[0];  // Apply H to qubit 0
        }
        if (sum == 3){
            z q[0];  // Apply Z to qubit 0
        }
        if (sum == 4){
            y q[0];  // Apply Y to qubit 0
        }
        """

program = OpenQASMProgram(source=qasm_source, inputs={})
simulator = BranchedSimulator()
result = simulator.run_openqasm(program, shots=1000)


print(result)


# Count measurement outcomes
measurements = result.measurements

print(measurements)

counter = Counter(["".join(measurement) for measurement in measurements])

print(counter)
