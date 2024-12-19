import pennylane as qml
from pennylane import numpy as np

def cost_hamiltonian(n_qubits):
    coeffs = [1.0 for index in range(n_qubits)]
    observables = [qml.PauliZ(index) for index in range(n_qubits)]
    return qml.Hamiltonian(coeffs, observables)

def mixer_hamiltonian(n_qubits):
    coeffs = [-1.0 for index in range(n_qubits)]
    observables = [qml.PauliX(index) for index in range(n_qubits)]
    return qml.Hamiltonian(coeffs, observables)

def QAOA_circuit(params, cost_h, mixer_h, n_qubits, p):
    for index in range(n_qubits):
        qml.Hadamard(wires=index)
    for layer in range(p):
        qml.evolve(cost_h, params[layer][0], num_steps=1)
        qml.evolve(mixer_h, params[layer][1], num_steps=1)

def cost_function(params, cost_h, mixer_h, n_qubits, p):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def circuit():
        QAOA_circuit(params, cost_h, mixer_h, n_qubits, p)
        return qml.expval(cost_h)
    return circuit()

def optimize_qaoa(n_qubits, p):
    if not isinstance(p, int):
        raise ValueError("The number of layers 'p' must be an integer.")
    cost_h = cost_hamiltonian(n_qubits)
    mixer_h = mixer_hamiltonian(n_qubits)
    params = np.random.uniform(0, np.pi, size=(p, 2))
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    max_iter = 50
    for step in range(max_iter):
        params = opt.step(lambda p: cost_function(p, cost_h, mixer_h, n_qubits, p), params)
        cost = cost_function(params, cost_h, mixer_h, n_qubits, p)
        print(f"Step {step+1}: Cost = {cost}")
    return params

n_qubits = 4
p = 2
optimal_params = optimize_qaoa(n_qubits, p)
print("Optimal QAOA Parameters:", optimal_params)