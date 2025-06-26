import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

matplotlib.use('TkAgg')
# Generate synthetic data
points = pd.read_csv('./dataset/Salary_dataset.csv')

# Gradient descent
m = 9640
b = 24000
learning_rate = 0.001
iterations = 50000
params = []

X = points.YearsExperience.values
Y = points.Salary.values

for _ in range(iterations):
    n = len(X)
    y_predicted = m * X + b
    
    error = Y - y_predicted

    m_gradient = (-2/n) * np.dot(X, error)
    b_gradient = -(2/n) * np.sum(error)

    m = m - learning_rate * m_gradient
    b = b - learning_rate * b_gradient
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient
    params.append((m, b))

# Plot setup
max_salary = points.Salary.max()
max_experience = points.YearsExperience.max()
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(points.YearsExperience, points.Salary, color='blue', label='Data Points')
line, = ax.plot([], [], 'r-', label='Regression Line')
ax.set_xlim(0, max_experience)
ax.set_ylim(0, max_salary + 10000)
ax.set_xlabel('YearsExperience')
ax.set_ylabel('Salary')
ax.set_title('Gradient Descent Iterations')
ax.legend()

def animate(i):
    m, b = params[i]
    x_vals = np.array([0, max_experience])
    y_vals = m * x_vals + b
    line.set_data(x_vals, y_vals)
    ax.set_title(f'Iteration {i + 1}: m={m:.2f}, b={b:.2f}')
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(params), interval=5, blit=False)
plt.show()
