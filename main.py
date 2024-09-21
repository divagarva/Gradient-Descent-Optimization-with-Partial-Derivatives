import numpy as np
import matplotlib.pyplot as plt

# Function to optimize: f(x, y) = x^2 + y^2
def function(x, y):
    return x**2 + y**2

# Partial derivatives of the function
def partial_derivatives(x, y):
    df_dx = 2 * x  # ∂f/∂x
    df_dy = 2 * y  # ∂f/∂y
    return np.array([df_dx, df_dy])

# Gradient descent algorithm
def gradient_descent(starting_point, learning_rate, iterations):
    points = [starting_point]
    current_point = starting_point

    for _ in range(iterations):
        grad = partial_derivatives(*current_point)
        next_point = current_point - learning_rate * grad
        points.append(next_point)
        current_point = next_point

    return np.array(points)

# Parameters
initial_point = np.array([4.0, 3.0])  # Starting point (x0, y0)
learning_rate = 0.1  # Step size
iterations = 50  # Number of iterations

# Perform gradient descent
path = gradient_descent(initial_point, learning_rate, iterations)

# Plotting the function surface and the path of gradient descent
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = function(X, Y)

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=30, cmap='viridis')
plt.plot(path[:, 0], path[:, 1], 'ro-', label='Gradient Descent Path')
plt.scatter(*initial_point, color='red', label='Starting Point')
plt.scatter(0, 0, color='blue', label='Minimum Point')
plt.title('Gradient Descent Optimization with Partial Derivatives')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()