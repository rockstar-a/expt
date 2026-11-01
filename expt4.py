# 4. AND, NAND, OR Gates - Neural Network
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Input combinations
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# Gate outputs
gates = {
    "AND":  np.array([[0],[0],[0],[1]]),
    "NAND": np.array([[1],[1],[1],[0]]),
    "OR":   np.array([[0],[1],[1],[1]])
}

for name, Y in gates.items():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(4, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X, Y, epochs=2000, verbose=0)

    pred = (model.predict(X, verbose=0) > 0.5).astype(int)

    print("\n", name, "Gate")
    print("Predicted Output:\n", pred)
    print("Accuracy:", model.evaluate(X, Y, verbose=0)[1])

    plt.plot(history.history['loss'])
    plt.title(name + " Gate - Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
