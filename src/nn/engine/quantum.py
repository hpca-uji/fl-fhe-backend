import tomli
import pennylane as qml

with open("pyproject.toml", "rb") as f:
    config = tomli.load(f)

num_qubits = config["tool"]["federated_learning"]["quantum"]["num-qubits"]

dev = qml.device("default.qubit", wires=num_qubits)
    
@qml.qnode(dev, interface='torch')
def quantum_net(inputs, weights, num_qubits=8):
    qml.AngleEmbedding(inputs, wires=range(num_qubits)) 
    qml.BasicEntanglerLayers(weights,wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
