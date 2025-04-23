import numpy as np

inputs = np.array([s
    [1, -1, 1],
    [-1, 1, -1],
    [1, 1, -1]
])

targets = np.array([
    [1],
    [-1],
    [1]
])

weights = np.zeros((inputs.shape[1], targets.shape[1]))

for i in range(len(inputs)):
    x = inputs[i].reshape(-1, 1)
    y = targets[i].reshape(-1, 1)
    weights += np.dot(x, y.T)

print("Trained Weights:\n", weights)

def predict(input_pattern):
    output = np.dot(input_pattern, weights)
    return np.where(output >= 0, 1, -1)

print("\nTesting:")
for i in range(len(inputs)):
    out = predict(inputs[i])
    print(f"Input: {inputs[i]} → Predicted Output: {out.ravel()}")
