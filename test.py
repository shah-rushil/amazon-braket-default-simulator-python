from braket.default_simulator.branched_simulator import BranchedSimulator
from collections import Counter, defaultdict
from braket.ir.openqasm import Program as OpenQASMProgram
from braket.devices import LocalSimulator

qasm_source = """
OPENQASM 3.0;
bit __bit_2__;
bit __bit_5__;
bit __bit_8__;
bit __bit_11__;
bit __bit_14__;
bit __bit_17__;
bit __bit_20__;
qubit[11] __qubits__;
bit[7] mcm = "0000000";
h __qubits__[10];

cnot __qubits__[10], __qubits__[9];

"""

program = OpenQASMProgram(source=qasm_source, inputs={})

python_result = LocalSimulator("braket_sv_branched_python").run(program, shots=1000).result()

print(python_result.measurement_counts)