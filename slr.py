import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.array([8, 10, 12, 14, 16, 18]).reshape(-1, 1)
Y = np.array([5, 7, 9, 11, 13, 15])

model = LinearRegression()
model.fit(X, Y)

m = model.coef_[0]
b = model.intercept_

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

Y_pred = model.predict(X)

accuracy = r2_score(Y, Y_pred)
print(f"Model Accuracy (R² Score): {accuracy:.2f}")

def predict_price(size):
    return model.predict(np.array([[size]]))[0]

sizes_to_predict = input("Enter pizza sizes (in inches) separated by commas: ")
sizes_to_predict = [float(size.strip()) for size in sizes_to_predict.split(',')]

for size in sizes_to_predict:
    predicted_price = predict_price(size)
    print(f"Predicted price for a {size}-inch pizza: ${predicted_price:.2f}")
