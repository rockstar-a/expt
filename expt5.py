# 4+5. Logic Gates

import numpy as np
from sklearn.neural_network import MLPClassifier

# Input data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Outputs for logic gates
gates = {
    "AND":  [0, 0, 0, 1],
    "NAND": [1, 1, 1, 0],
    "OR":   [0, 1, 1, 1],
    "NOR":  [1, 0, 0, 0],
    "XOR":  [0, 1, 1, 0]
}

# Create and train neural network
def train_model(Y):

    model = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation='logistic',
        solver='lbfgs',
        max_iter=2000,
        random_state=42
    )

    model.fit(X, Y)
    return model


# Train and predict all gates
for name, Y in gates.items():

    model = train_model(Y)

    print("\n", name, "Gate:")
    print("Predicted Output:", model.predict(X))
