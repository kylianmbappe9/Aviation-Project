import pennylane as qml
from pennylane import numpy as np
import numpy as np

def create_QUBO(fuel_consumption, distances, fuel_price):
    #create a matrix with our n_flights being the row part 
    #and n columns being how much fuel consumed for these two flights 
    #when they used route 1
    #example [2x2 matrix] (NOT ACTUAL VALUE)
    # fuel consumption = [(cell-11) -> 3.0, (cell-12) -> 2.5, (cell-21) -> 2.8, (cell-22) -> 3.8]
    # 3.0 and 2.5 will be how much fuel can be consumed (predicted not actual values)
    #by that one specific flight (call it flight 1)
    #3.0 and 2.8 denote how much fuel can route 1 be consumed for flight 1 and flight 2
    #we want to observe which OPEN FLIGHT (AVAILABLE ATM)
    #will take the (CONFIRMED OPTIMAL ROUTE) and we will output that
    n_flights, n_routes = fuel_consumption.shape
    #initialize with all zeros
    QUBO = np.zeros((n_flights * n_routes, n_flights * n_routes))  # Fixed matrix initialization
    #now we want to iterate through every flight (whether available or booked)
    #we want to count how many planes exist at that moment for that flying destination
    for i in range(n_flights):
        for j in range(n_routes):
            index = i * n_routes + j  # Corrected indexing for flattened matrix
            QUBO[index, index] = fuel_consumption[i, j] * distances[i, j] * fuel_price
    return QUBO

#test output for the QUBO matrix
#this is just a matrix; however, next we will implement the 
#model that actually processes this matrix
fuel_consumption = np.array([[3.0, 2.8], [2.5, 3.2]])
distances = np.array([[500, 600], [400, 500]])
fuel_price = 1.5
n_flights, n_routes = fuel_consumption.shape

QUBO = create_QUBO(fuel_consumption, distances, fuel_price)  # Correct QUBO generation


def quantum_model(QUBO, wires):
    num_qubits = len(wires)
    
    #input matrix for available flights
    for i in range(num_qubits):
        qml.RX(0.01, wires=i)  # Small RX rotation to initialize states
    
    #this is for the flight and route with the 
    #penalty matrix
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if QUBO[i, j] != 0:  # Add interactions for non-zero QUBO elements
                qml.IsingZZ(QUBO[i, j], wires=[i, j])

    for i in range(num_qubits):
        qml.RY(0.01, wires=i)  # Final layer with RY rotations


def cost_function(params, QUBO, wires):
    """
    Cost function to evaluate the performance of the quantum circuit.
    This integrates the QUBO matrix, variational parameters, and the qubit wires.

    Parameters:
    - params: Variational parameters used to optimize the quantum circuit.
              These are updated iteratively to minimize the cost.
    - QUBO: Quadratic Unconstrained Binary Optimization (QUBO) matrix that encodes the fuel cost problem.
    - wires: List of qubits (decision variables), where each qubit corresponds to a route for a flight.
    """

    # Determine the number of qubits (one qubit per decision variable)
    num_qubits = len(wires)

    # Set up the quantum device (simulator in this case)
    # "default.qubit" is a PennyLane built-in simulator for qubit-based circuits
    # In real-world scenarios, this can be replaced with a hardware backend like Qiskit or D-Wave
    dev = qml.device("default.qubit", wires=num_qubits)

    # Define the quantum circuit
    @qml.qnode(dev)
    def circuit(QUBO, wires):
        """
        Quantum circuit that implements the QUBO problem using variational quantum gates.
        
        - Encodes the QUBO matrix into the quantum circuit.
        - Measures the expectation values of Pauli-Z for each qubit.
        """

        # Encode the QUBO matrix into the quantum circuit using the quantum_model function
        quantum_model(QUBO, wires)

        # Measure the expectation value of Pauli-Z for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]  # Return all expectation values

    # The cost function integrates:
    # - QUBO encoding in the quantum circuit
    # - Measurement of expectation values
    # - Classical optimization to iteratively refine variational parameters (params)

    expectations = circuit(QUBO, wires)
    cost = 0
    for i in range(num_qubits):
        cost += QUBO[i, i] * expectations[i]
        for j in range(i+1, num_qubits):
            cost += QUBO[i, j] * expectations[i] * expectations[j]
    
    return cost


# Simulated input: fuel consumption, distances, and fuel price
np.random.seed(42)
n_flights = 2  # Example: 2 flights
routes_per_flight = 3  # 3 routes per flight

# Randomized data for testing
fuel_consumption = np.random.uniform(2.0, 4.0, size=(n_flights, routes_per_flight))
distances = np.random.randint(300, 800, size=(n_flights, routes_per_flight))
fuel_price = 1.5  # Cost per liter of fuel

# Generate QUBO matrix
QUBO = create_QUBO(fuel_consumption, distances, fuel_price)
num_qubits = QUBO.shape[0]
wires = list(range(num_qubits))  # Qubits for each flight-route pair
init_params = np.random.uniform(0, np.pi, size=num_qubits)

# Minimize cost function using classical optimization
from scipy.optimize import minimize
result = minimize(cost_function, init_params, args=(QUBO, wires), method="COBYLA")

# Display results
print("\n### Optimization Results ###")
print("Fuel Consumption (liters/km):\n", fuel_consumption)
print("Distances (km):\n", distances)
print("Fuel Price ($/liter):", fuel_price)
print("\nOptimal Parameters:", result.x)
print("Optimal Cost (Fuel):", result.fun)
